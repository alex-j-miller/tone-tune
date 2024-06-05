import streamlit as st
import pandas as pd
import html
import re

model_name = 'google/gemma-1.1-2b-it'

formal_options = [
    'More aggressive',
    'More kind',
    'More simple',
    'More complex',
    'Easier to read',
    'More flirtatious',
    'Grammatically correct',
    'More informal',
    'Business casual',
    'More sarcastic',
    'Less sarcastic',
    'Like Eric Cartman'
]

@st.cache_resource
def get_tokenizer(model_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name).from_pretrained(model_name)

@st.cache_resource
def get_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    print(f"Loaded model, {model.num_parameters():,d} parameters.")
    return model

doc = st.text_area("Document", "This is a document that I would like to have rewritten to be more concise.")
updated_doc = st.text_area("Updated Doc", help="Your edited document. Leave this blank to use your original document.")


def get_spans_local(prompt, doc, updated_doc):
    import torch
    
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)


    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\n{doc}",
        },
    ]

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
    assert len(tokenized_chat.shape) == 1

    if len(updated_doc.strip()) == 0:
        updated_doc = doc
    updated_doc_ids = tokenizer(updated_doc, return_tensors='pt')['input_ids'][0]
    joined_ids = torch.cat([tokenized_chat, updated_doc_ids[1:]])

    # Call the model
    with torch.no_grad():
        logits = model(joined_ids[None].to(model.device)).logits[0].cpu()

    spans = []
    length_so_far = 0
    for idx in range(len(tokenized_chat), len(joined_ids)):
        probs = logits[idx - 1].softmax(dim=-1)
        token_id = joined_ids[idx]
        token = tokenizer.decode(token_id)
        token_loss = -probs[token_id].log().item()
        most_likely_token_id = probs.argmax()
        print(idx, token, token_loss, tokenizer.decode(most_likely_token_id))
        spans.append(dict(
            start=length_so_far,
            end=length_so_far + len(token),
            token=token,
            token_loss=token_loss,
            most_likely_token=tokenizer.decode(most_likely_token_id)
        ))
        length_so_far += len(token)
    return spans

emojis = """
:grinning:
:grin:
:joy:
:smiley:
:smile:
:sweat_smile:
:laughing:
:satisfied:
:innocent:
:smiling_imp:
:wink:
:blush:
:yum:
:relieved:
:heart_eyes:
:sunglasses:
:smirk:
:neutral_face:
:expressionless:
:unamused:
:sweat:
:pensive:
:confused:
:confounded:
:kissing:
:kissing_heart:
:kissing_smiling_eyes:
:kissing_closed_eyes:
:stuck_out_tongue:
:stuck_out_tongue_winking_eye:
:stuck_out_tongue_closed_eyes:
:disappointed:
:worried:
:angry:
:rage:
:cry:
:persevere:
:triumph:
:disappointed_relieved:
:frowning:
:anguished:
:fearful:
:weary:
:sleepy:
:tired_face:
:grimacing:
:sob:
:face_exhaling:
:open_mouth:
:hushed:
:cold_sweat:
:scream:
:astonished:
:flushed:
:sleeping:
:face_with_spiral_eyes:
:dizzy_face:
"""

def perform_action(action, text):
    
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    messages = [
        {
            "role": "user",
            "content": f"Please {action} the following text:\n\n{text}",
        },
    ]
    
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    summary_ids = model.generate(tokenized_chat, max_new_tokens=500, num_beams=2, early_stopping=True)
    summary_ids = summary_ids[:, tokenized_chat.shape[1]:]
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def pick_emoji(text):
    
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    messages = [
        {
            "role": "user",
            "content": f"Given the tone of the text at the end, select a single emoji shortcode from the following list and output it: \n\n{emojis}\n\n{text}",
        },
    ]
    
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    summary_ids = model.generate(tokenized_chat, max_new_tokens=500, num_beams=2, early_stopping=True)
    summary_ids = summary_ids[:, tokenized_chat.shape[1]:]
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    q = re.search(":.*:", summary)
    o = q.group() if q != None else ":frowning:"
    p = re.search(o, emojis)
    return p.group() if p != None else ":angry:"

mini_outs = ""
emoji = ""
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

with col1:
    if st.button("Summarize"):
        mini_outs = perform_action("summarize", doc)

with col2:
    if st.button("Identify Tone"):
        mini_outs = perform_action("identify the tone of", doc)
        emoji = pick_emoji(doc)

with col3:
    if st.button("Suggest +"):
        mini_outs = perform_action("(in two or 3 sentences) suggest some ways to expand upon", doc)

with col4:
    if st.button("Suggest -"):
        mini_outs = perform_action("(in two or 3 sentences) suggest some ways to shorten", doc)

with col5:
    if st.button("Response"):
        mini_outs = perform_action("describe a likely response from the person recieving", doc)

if mini_outs != "":
    st.write(mini_outs)

if emoji != "":
    st.write(emoji)

st.markdown("***")

prompt = "Rewrite this document to be " + st.selectbox("Rewrite this document to be: ", formal_options)

spans = get_spans_local(prompt, doc, updated_doc)

if len(spans) < 2:
    st.write("No spans found.")
    st.stop()

highest_loss = max(span['token_loss'] for span in spans[1:])
for span in spans:
    span['loss_ratio'] = span['token_loss'] / highest_loss

html_out = ''
for span in spans:
    is_different = span['token'] != span['most_likely_token']
    val = 255.0 * float(span['loss_ratio'])
    html_out += '<span style="color: {color}" title="{title}">{orig_token}</span>'.format(
        color="rgb(" + str(val + 100) + ", 0, " + str(355.0-val) + ")" if is_different else "white",
        title=html.escape(span["most_likely_token"]).replace('\n', ' '),
        orig_token=html.escape(span["token"]).replace('\n', '<br>')
    )
html_out = f"<p>{html_out}</p>"

st.write(html_out, unsafe_allow_html=True)
st.dataframe(pd.DataFrame(spans)[['token', 'token_loss', 'most_likely_token', 'loss_ratio']], use_container_width=True)
