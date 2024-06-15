# %% imports
import random

from datasets import load_dataset
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from diffusers.utils import (
    make_image_grid,
)  # i imported diffusers cause im too lazy to define make_image_grid function
from faker import Faker
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

# %% fake sensor data generation
fake = Faker()


def decimal_to_dms(degrees):
    degrees = float(degrees)
    is_positive = degrees >= 0
    degrees = abs(degrees)
    d = int(degrees)
    m = int((degrees - d) * 60)
    s = (degrees - d - m / 60) * 3600
    return d, m, s, is_positive


def format_latitude(lat):
    d, m, s, is_positive = decimal_to_dms(lat)
    direction = "N" if is_positive else "S"
    return f"{direction}{d}°{m}'{s:.0f}\""


def format_longitude(lon):
    d, m, s, is_positive = decimal_to_dms(lon)
    direction = "E" if is_positive else "W"
    return f"{direction}{d}°{m}'{s:.0f}\""


def generate_sensor_data():
    latitude = format_latitude(fake.latitude())
    longitude = format_longitude(fake.longitude())
    date = fake.date(pattern="%d.%m.%Y")
    time = fake.time(pattern="%H:%M:%S")
    speed = f"{random.uniform(0, 100):.1f} mph"
    return {
        "latitude": latitude,
        "longitude": longitude,
        "date": date,
        "time": time,
        "speed": speed,
    }


data = generate_sensor_data()
print(data)


# %% ollama loading
llm = Ollama(
    model="llama3"
)  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3 `

llm.invoke("Tell me a joke")


# %% load ph3 model. it take image and return unstructured text
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)  # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
messages = [
    {
        "role": "user",
        "content": "<|image_1|>\nWhat is date and time and exact location and speed in the image?",
    },
]

# %% define llm chain. it take unstructured text and return structured text(dictionary)
response_schemas = [
    ResponseSchema(
        name="latitude",
        # description="latitude of the gps location in degrees, minutes, and seconds",
        description="latitude of the gps location in DD°MM'SS",
    ),
    ResponseSchema(
        name="longitude",
        # description="longitude of the gps location in degrees, minutes, and seconds",
        description="longitude of the gps location in DD°MM'SS",
    ),
    ResponseSchema(name="date", description="date in DD.MM.YYYY format"),
    ResponseSchema(name="time", description="time in HH:MM:SS format"),
    ResponseSchema(name="speed", description="speed in mph"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="summarize input text as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | output_parser


# %% load dataset
dataset = load_dataset("segments/sidewalk-semantic")
train_ds = dataset["train"]
train_ds = train_ds.shuffle()


train_ds[0]["pixel_values"]


# %% test random 10 samples

concat_images = []
for example in train_ds.take(10):
    raw_image = example["pixel_values"]

    # generate sensor data
    data = generate_sensor_data()

    # convert generated sensor data to  multiline text
    text = """{date} {time}
    {latitude} {longitude} {speed}""".format(
        **data
    )

    # image.crop((600, 850, 1250, 1050))
    # draw it onto image

    annotated_image = raw_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    position = (700, 950)  # for fhd image
    font_color = (255, 255, 255)

    draw.text(
        position,
        text,
        fill=font_color,
        font_size=30,
        stroke_width=0,  # you can make it bolder with a bigger number
        stroke_fill="white",
    )

    # crop text overlayed portion of image
    image = annotated_image.crop((600, 850, 1250, 1050))

    # inference with phi3 model. it return unstructured text

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    generation_args = {
        "max_new_tokens": 500,
        # "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )

    # remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    res = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(res)

    # now we have unstructured text. we need to convert it to structured text
    multilines = []

    for _ in range(10):  # sometimes it give error. so we try 10 times here
        try:
            record = chain.invoke({"question": res})
            multilines.append("200")
            break
        except Exception:
            continue
    else:
        multilines.append("400")

    for key in ["date", "time", "latitude", "longitude", "speed"]:
        val = record.get(key, "")

        if val == data[key]:
            multilines.append(f"{key}:")
        else:
            multilines.append(f"{key}: {val} != {data[key]}")

    # match하냐 안하냐로 visualization 하면 되겠다.
    annotated_image = Image.new("RGB", image.size, "white")
    draw = ImageDraw.Draw(annotated_image)
    draw.text( (0,0),  "\n".join(multilines), fill = (255,0,0), font_size=20)
    concat_image = make_image_grid([image, annotated_image], 1, 2)
    concat_images.append(concat_image)

    if len(concat_images) == 5:
        break


# %%
make_image_grid(concat_images, len(concat_images), 1)




