import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


from imagetocode import Model, test, build_vocab, encode, tokenize


import pickle

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(vocab_size=len(vocab)).to(device)


model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


st.title("Image-to-Code Generator")
st.caption("Upload a UI image to generate code")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    pil_img = Image.open(uploaded_image).convert("RGB")
    image_tensor = transform(pil_img).to(device)

    with st.spinner("Generating code..."):
        generated_code = test(model, image_tensor, vocab)

    st.subheader("Generated Code")
    st.code(generated_code, language="html")
