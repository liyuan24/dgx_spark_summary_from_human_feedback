from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    model = "Qwen/Qwen2.5-0.5B"
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype="auto", device_map="auto"
    )
    input_text = """SUBREDDIT: r/tifu

TITLE: TIFU by forgetting my lube in the shower

POST: So I'm sitting in the living room with my then GF not long ago and my roommate (Carl with the slicked back hair) comes in from the bathroom to ask me where the little black bottle in the shower went. Confused, I looked back at him and told him I didn't know what he was talking about. This is about what happened next:

>Carl: Are you sure it wasn't yours? It appeared in the bathroom the other day and then today it's gone.

>Me: Still not sure. Why do you ask?

>Carl: Aww...damn. Whatever that shit was it was awesome. It was like this conditioner or something that kept my hair super slick alll day long. It was crazy, not even water could get it out. It worked so much better than the hair stuff I use right now. Maybe Jenny (his GF) knows where it went. Are you sure you don't know?

At this point I have figured out that the little black bottle to which he kept referring was my bottle of lube. I glanced over at my GF and all the color had drained from her face and she was trying not to give it away that she knew. With the straightest face I could muster I told him that I still had no idea. I haven't told him to this day.

TL;DR:"""
    tokenizer = AutoTokenizer.from_pretrained(model)
    inputs = tokenizer(input_text, return_tensors="pt").to(qwen_model.device)
    outputs = qwen_model.generate(**inputs, max_new_tokens=63)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

