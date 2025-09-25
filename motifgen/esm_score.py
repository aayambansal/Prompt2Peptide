# --- esm_score.py ---
import torch
import esm

# small & fast
_MODEL_NAME = "esm2_t6_8M_UR50D"

@torch.no_grad()
def esm_avg_loglik(seq:str)->float:
    """Calculate average log-likelihood of sequence using ESM-2 model"""
    model, alphabet = esm.pretrained.load_model_and_alphabet(_MODEL_NAME)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    if torch.cuda.is_available():
        model = model.cuda(); tokens = tokens.cuda()
    out = model(tokens, repr_layers=[], return_contacts=False)
    # negative log-likelihood per token
    toks = tokens[:,1:-1]
    lprobs = out["logits"].log_softmax(-1)
    gather = torch.gather(lprobs[:, :-1, :], 2, toks[:,1:].unsqueeze(-1)).squeeze(-1)
    return float(gather.mean().cpu())
