import asyncio
import logging
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.filters import Command
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# 🔑 Telegram token
TELEGRAM_TOKEN = "8436321772:AAGFWjMkcHmeGGMYCPghlzfm7AYBhwp-GR0"

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# 🔹 Inline tugmalar
model_buttons = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="BlenderBot 90M", callback_data="model_blender")],
        [InlineKeyboardButton(text="DialoGPT-medium", callback_data="model_dialogpt")],
        [InlineKeyboardButton(text="MT5-small (O‘zbekcha)", callback_data="model_mt5")]
    ]
)

# 🔹 Foydalanuvchilarning tanlagan modeli va cache
user_models = {}
user_cache = {}

# 🔹 Modellarni yuklash
print("Modellarni yuklash... Iltimos kuting.")

# BlenderBot 90M
blender_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-90M", cache_dir="models/blenderbot")
blender_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-90M", cache_dir="models/blenderbot")

# DialoGPT-medium
dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir="models/DialogGPT")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", cache_dir="models/DialogGPT")

# MT5-small (ko‘p tilli, o‘zbekcha)
mt5_small_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir="models/mt5-small")
mt5_small_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", cache_dir="models/mt5-small")

print("Modelar yuklandi ✅")

# 🔹 /start yoki /help komandasi
@dp.message(Command(commands=["start", "help"]))
async def send_welcome(message: types.Message):
    await message.reply(
        "Salom! Men AI yordamchi botman.\n"
        "Modelni tanlang va xabaringizni yuboring:",
        reply_markup=model_buttons
    )

# 🔹 Callback: model tanlash
@dp.callback_query()
async def process_model(callback: types.CallbackQuery):
    selected_model = callback.data
    user_models[callback.from_user.id] = selected_model

    # Cache ni tozalash
    user_cache[callback.from_user.id] = {}

    await callback.answer(text=f"Model tanlandi: {callback.data.split('_')[1]}")
    await callback.message.reply("Endi xabaringizni yozing, AI javob beradi...")

# 🔹 Foydalanuvchi xabarini AI ga yuborish
@dp.message()
async def ai_reply(message: types.Message):
    user_id = message.from_user.id
    selected_model = user_models.get(user_id)

    if not selected_model:
        await message.reply("Iltimos, avval modelni tanlang:", reply_markup=model_buttons)
        return

    text = message.text.strip()
    if not text:
        await message.reply("Uzr, bo‘sh xabar yuborildi. Iltimos, savol yozing.")
        return

    # Cache tekshirish
    if user_id not in user_cache:
        user_cache[user_id] = {}
    if text in user_cache[user_id]:
        await message.reply(user_cache[user_id][text] + " (cached)")
        return

    answer = ""
    try:
        # 🔹 Model tanlash va javob olish
        if selected_model == "model_blender":
            tokenizer = blender_tokenizer
            model = blender_model
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        elif selected_model == "model_dialogpt":
            tokenizer = dialogpt_tokenizer
            model = dialogpt_model
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        elif selected_model == "model_mt5":
            tokenizer = mt5_small_tokenizer
            model = mt5_small_model

            # MT5 uchun chat prompti (o‘zbek tilida javob)
            prompt = f"Generate a conversational answer in Uzbek: {text}"
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    repetition_penalty=2.0,
                    early_stopping=True,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if not answer:
            answer = "Uzr, hozir javob topilmadi. Iltimos qayta urinib ko‘ring."

    except Exception as e:
        print(f"Xato yuz berdi: {e}")
        answer = "Uzr, xatolik yuz berdi. Iltimos qayta urinib ko‘ring."

    # Cache ga saqlash
    user_cache[user_id][text] = answer
    await message.reply(answer)

# 🔹 Botni ishga tushirish
if __name__ == "__main__":
    print("🤖 Bot ishga tushdi...")
    asyncio.run(dp.start_polling(bot))
