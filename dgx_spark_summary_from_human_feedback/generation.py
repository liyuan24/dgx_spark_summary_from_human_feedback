import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .sft import SFTConfig

TEST_PROMPT = """TITLE: TIFU by forgetting my lube in the shower

POST: So I'm sitting in the living room with my then GF not long ago and my roommate (Carl with the slicked back hair) comes in from the bathroom to ask me where the little black bottle in the shower went. Confused, I looked back at him and told him I didn't know what he was talking about. This is about what happened next:

>Carl: Are you sure it wasn't yours? It appeared in the bathroom the other day and then today it's gone.  
>Me: Still not sure. Why do you ask?  
>Carl: Aww...damn. Whatever that shit was it was awesome. It was like this conditioner or something that kept my hair super slick all day long. It was crazy, not even water could get it out. It worked so much better than the hair stuff I use right now. Maybe Jenny (his GF) knows where it went. Are you sure you don't know?

At this point I have figured out that the little black bottle to which he kept referring was my bottle of lube. I glanced over at my GF and all the color had drained from her face and she was trying not to give it away that she knew. With the straightest face I could muster I told him that I still had no idea. I haven't told him to this day.

TL;DR:"""

TEST_PROMPT_2 = """SUBREDDIT: r/relationships

TITLE: Me 28 F with guy I'm dating 30 M - 1 month, Is it okay to ask if everything is okay or am I being pushy?

POST: I have been dating this guy for 1 month now and everything was great until last Sunday when I noticed he wasn't responding to my text with the same flirty, cute, enthusiastic text that he usually would. He used to call me baby, babe but since last Sunday he has said none of those words and texting has been less and communication has been less.

He invited me over last night to watch a movie so I went over and I think we had one conversation the whole night, it was how my day was. He never even tried to start a conversation after the movie and he has never asked if he could get high in front of me until today which I was totally fine with but I thought it was weird that the whole time we hung out before he never smoked in front of me. I am not sure if he's getting comfortable or he just doesn't care anymore?

Anyways, as he was walking me out to my car I asked him if everything was okay because he has been acting different. His only response was, "Yes, I'm fine" and then it got awkward and I left in my car.

Do you think I am being pushy or too clingy asking that question? I regret asking it right after I asked it because it makes me feel like I have low self-esteem for this relationship. Maybe I just worry too much but it has been eating at me.

TL;DR:"""


def load_hf_checkpoint(checkpoint_path, dtype="bfloat16"):
    """
    Load a fine-tuned checkpoint and return the model and tokenizer.
    """
    print("Load HF checkpoint from: ", checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def load_torch_checkpoint(
    checkpoint_path, model_name, device_map="auto", dtype="bfloat16"
):
    """
    Load a fine-tuned checkpoint and return the model and tokenizer.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt file)
        model_name: Original model name (for tokenizer and config)
        device_map: Device mapping strategy for model loading
        dtype: Model dtype (bfloat16, float16, or float32)

    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
        checkpoint_info: Dictionary with checkpoint metadata
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    # weights_only=False needed for PyTorch 2.6+ when loading checkpoints with custom objects like Qwen2Config
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Load model
    print(f"Loading model from: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Load the fine-tuned weights
    print("Loading fine-tuned weights from checkpoint...")
    model.load_state_dict(checkpoint["model"], strict=True)

    # Move model to eval mode for inference
    model.eval()

    print("Checkpoint loaded successfully!")
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=63,
    temperature=0.0,
):
    """
    Generate text from a prompt using the loaded model.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling

    Returns:
        generated_text: Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text, outputs[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a fine-tuned checkpoint and generate text"
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        default=False,
        help="Use base model for text generation",
    )
    parser.add_argument(
        "--use_hf_checkpoint",
        action="store_true",
        default=True,
        help="Use HF checkpoint for text generation",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Original model name (for tokenizer and config)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=63,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_base_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True, dtype=args.dtype, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True
        )
    else:
        # Load checkpoint
        assert args.checkpoint_path is not None, "Checkpoint path is required"
        if args.use_hf_checkpoint:
            model, tokenizer = load_hf_checkpoint(args.checkpoint_path, args.dtype)
        else:
            model, tokenizer = load_torch_checkpoint(
                args.checkpoint_path, args.model_name, args.dtype
            )

    # Generate text
    print(f"\n{'='*70}")
    print("Generating text with prompt")
    print(f"{'='*70}\n")

    inputs = tokenizer(TEST_PROMPT_2, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    # Decode generated text
    generated_text = tokenizer.decode(outputs[0][input_length:])

    print(f"Generated text:\n{generated_text}\n")
    print(f"Generated tokens:\n{outputs[0][input_length:]}\n")
