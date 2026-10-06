"""FP32 diagnostics; solver boundary samples (including tau=0 and tau=1)."""
import csv
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from src.models.projected_node import ProjectedNODE
from src.models.structured_node import StructuredNODE


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def diagnose(model, tokens, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    x_all = model.embedding(tokens)
    if not hasattr(model, "initial_state"):
        (output/"diagnostics.json").write_text(json.dumps({"model": "transformer", "ode_diagnostics": "not applicable"}, indent=2))
        return
    state = model.initial_state(tokens.size(0), x_all.device, x_all.dtype)
    rows, token_rows, coefficients, full_trajectories, memories = [], [], [], [], []
    for pos, x in enumerate(x_all.unbind(1)):
        h = model.memory(state)
        if hasattr(model, "field"):
            if isinstance(model, ProjectedNODE):
                start = model.project_in(torch.cat((h, x), -1))
                end, trajectory = model.latent_solve(start, x, True)
                next_state = model.project_out(end)
                field = lambda s: model.field(s, x if model.conditioned else None)
                space = "latent_s"
            else:
                start = h
                end, trajectory = model.solve(h, x, True)
                next_state = end
                field = lambda s: model.field(s, x)
                space = "memory_h"
            full_trajectories.append(torch.stack(trajectory, dim=1).cpu())
            cosine = F.cosine_similarity(start, end, dim=-1)
            displacement = (end-start).norm(dim=-1)
            for k, point in enumerate(trajectory):
                derivative = field(point)
                for example in range(tokens.size(0)):
                    rows.append(dict(example=example, position=pos, token=int(tokens[example, pos]),
                                     tau=k/model.cfg.steps_per_token, state_space=space,
                                     hidden_norm=float(point[example].norm()),
                                     derivative_norm=float(derivative[example].norm()),
                                     cosine_start_end=float(cosine[example]), update_norm=float(displacement[example])))
            if isinstance(model, StructuredNODE):
                a, _ = model.field.coefficients(x)
                for example in range(a.size(0)):
                    for channel in range(a.size(1)):
                        av = float(a[example, channel])
                        q = av/model.cfg.steps_per_token
                        method = model.cfg.integrator
                        stability = 1+q if method == "euler" else 1+q+q*q/2 if method == "heun" else 1+q+q*q/2+q**3/6+q**4/24
                        discrete_amplification = stability**model.cfg.steps_per_token
                        coefficients.append(dict(example=example, position=pos, channel=channel, a=av,
                                                 exp_a=float(torch.exp(a[example, channel])),
                                                 timescale=-1/av if av < 0 else "", discrete_amplification=discrete_amplification, discretely_stable=abs(discrete_amplification) < 1))
        else:
            next_state = model.transition(x, state)
        after = model.memory(next_state)
        memories.append(after.cpu())
        for example in range(tokens.size(0)):
            token_rows.append(dict(example=example, position=pos, token=int(tokens[example, pos]),
                                   hidden_norm=float(after[example].norm()),
                                   update_norm=float((after-h)[example].norm()),
                                   cosine_start_end=float(F.cosine_similarity(h[example:example+1], after[example:example+1], dim=-1)[0])))
        state = next_state
    if full_trajectories:
        torch.save(torch.stack(full_trajectories, dim=1), output/"internal_states.pt")
    if memories:
        torch.save(torch.stack(memories, dim=1), output/"token_memories.pt")
    write_csv(output/"internal_trajectory.csv", rows)
    write_csv(output/"token_dynamics.csv", token_rows)
    write_csv(output/"structured_coefficients.csv", coefficients)
    report = {"precision": "FP32", "step_sampling": "all solver boundaries; derivative probes are extra NFE excluded from benchmark",
              "zero_start_cosine": "torch convention is zero when one vector is zero"}
    if hasattr(model, "field"):
        dimension = model.cfg.latent_dim if isinstance(model, ProjectedNODE) else model.cfg.hidden_dim
        vocab = model.embedding.num_embeddings
        vectors = model.embedding(torch.arange(vocab, device=tokens.device))
        gen = torch.Generator(device=tokens.device).manual_seed(17)
        cosine_matrices, norms, distances = [], [], []
        for fixed in (torch.zeros(dimension, device=tokens.device), torch.randn(dimension, device=tokens.device, generator=gen)):
            fixed = fixed.expand(vocab, -1)
            conditioned = not isinstance(model, ProjectedNODE) or model.conditioned
            derivatives = model.field(fixed, vectors if conditioned else None)
            norms.append(derivatives.norm(dim=-1).cpu())
            distances.append(torch.cdist(derivatives, derivatives).cpu())
            normalized = F.normalize(derivatives, dim=-1)
            cosine_matrices.append((normalized @ normalized.T).cpu())
        matrices = torch.stack(cosine_matrices)
        torch.save(matrices, output/"token_field_cosines.pt")
        offdiag = ~torch.eye(vocab, dtype=torch.bool)
        values = matrices[:, offdiag]
        report.update(token_field_cosine_mean=float(values.mean()), token_field_cosine_std=float(values.std()),
                      token_field_cosine_min=float(values.min()),
                      token_field_norm_mean=float(torch.stack(norms).mean()),
                      token_field_zero_fraction=float((torch.stack(norms) == 0).float().mean()),
                      token_field_pairwise_l2_mean=float(torch.stack(distances)[:, offdiag].mean()), fixed_states=["zero", "seed17_normal"],
                      includes_reserved_tokens=True)
    if coefficients:
        a = torch.tensor([r["a"] for r in coefficients])
        expa = a.exp()
        negative = a[a < 0]
        report.update(a_min=float(a.min()), a_max=float(a.max()), a_mean=float(a.mean()),
                      exp_a_min=float(expa.min()), exp_a_max=float(expa.max()),
                      negative_fraction=float((a < 0).float().mean()),
                      discretely_stable_fraction=sum(r["discretely_stable"] for r in coefficients)/len(coefficients),
                      max_discrete_amplification=max(abs(r["discrete_amplification"]) for r in coefficients),
                      time_constant_median=float((-1/negative).median()) if len(negative) else None,
                      note="Negative continuous eigenvalues do not guarantee stability of the configured discrete solver")
    report["finite_trajectory"] = all(torch.isfinite(torch.tensor(r["hidden_norm"])) and torch.isfinite(torch.tensor(r["derivative_norm"])) for r in rows)
    (output/"diagnostics.json").write_text(json.dumps(report, indent=2))
    return report
