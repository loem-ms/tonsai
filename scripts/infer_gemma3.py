import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 3 text generation inference")
    parser.add_argument("--model", type=str, required=True, help="Local path or HF model id")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply chat template with user role")
    parser.add_argument(
        "--chat-template-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Tokenizer source to borrow chat template from when model tokenizer has none",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_prompt(tokenizer, prompt_text: str, use_chat_template: bool, chat_template_model: str) -> str:
    if not use_chat_template:
        return prompt_text

    if getattr(tokenizer, "chat_template", None) is None:
        try:
            template_tokenizer = AutoTokenizer.from_pretrained(chat_template_model)
            tokenizer.chat_template = getattr(template_tokenizer, "chat_template", None)
        except Exception as exc:
            print(
                f"[warn] Failed to load chat template tokenizer '{chat_template_model}': {exc}; using raw prompt.",
                flush=True,
            )
            return prompt_text

    if getattr(tokenizer, "chat_template", None) is None:
        print(
            "[warn] No chat template found on tokenizer and fallback template could not be loaded; using raw prompt.",
            flush=True,
        )
        return prompt_text

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    prompt = _build_prompt(
        tokenizer=tokenizer,
        prompt_text=args.prompt,
        use_chat_template=args.use_chat_template,
        chat_template_model=args.chat_template_model,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = not args.no_sample
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": args.num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    full_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    new_token_ids = outputs[:, input_len:]
    new_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

    for i, (new_text, full_text) in enumerate(zip(new_texts, full_texts), start=1):
        print(f"\n=== Completion {i} ===")
        if new_text.strip():
            print(new_text)
        else:
            print("[warn] Model returned empty new text. Showing full decoded sequence instead:")
            print(full_text)


if __name__ == "__main__":
    main()
