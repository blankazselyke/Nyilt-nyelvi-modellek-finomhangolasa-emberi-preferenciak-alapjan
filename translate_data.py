import os
import pandas as pd
import asyncio
from openai import AsyncOpenAI
from datasets import load_dataset

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

translated_data = []
tasks = []

async def translate_text(text):
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Please translate the following text to Hungarian exactly as written, without changing or interpreting the meaning. Only provide the translated text, and ensure the output matches the structure of the input. Text: {text}",
            }
        ],
        model="gpt-4",
    )
    return response.choices[0].message.content

async def process_entry(entry):
    translated_system = await translate_text(entry["system"])
    translated_input = await translate_text(entry["input"])
    translated_chosen = await translate_text(entry["chosen"])
    translated_rejected = await translate_text(entry["rejected"])

    translated_entry = {
        "system": translated_system,
        "input": translated_input,
        "chosen": translated_chosen,
        "rejected": translated_rejected,
    }

    translated_data.append(translated_entry)  
    print(translated_entry)
    return translated_entry

async def main():
    dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")

    dataset = dataset.filter(
        lambda r:
        r["status"] != "tie" and
        r["chosen_score"] >= 8 and
        not r["in_gsm8k_train"]
    )

    counter = 0

    for entry in dataset:
        counter += 1
        print(f"Processing entry {counter}")
        tasks.append(process_entry(entry))

        if counter % 10 == 0:
            translated_batch = await asyncio.gather(*tasks)
            df = pd.DataFrame(translated_data)
            df.to_csv(f'D:\\DPO adatok\\translated_dataset_{counter // 10}.csv', sep=';', index=False, encoding='utf-8-sig')
            tasks.clear()

    if tasks:
        translated_batch = await asyncio.gather(*tasks)
        df = pd.DataFrame(translated_data)
        df.to_csv(f'D:\\DPO adatok\\translated_dataset_{(counter // 10) + 1}.csv', sep=';', index=False, encoding='utf-8-sig')

asyncio.run(main())