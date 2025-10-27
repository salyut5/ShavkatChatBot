import logging
import torch
import os
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.filters import Command
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from aiohttp import web

# 🔑 Muhit o'zgaruvchilari (Render uchun)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8436321772:AAGFWjMkcHmeGGMYCPghlzfm7AYBhwp-GR0")
WEBHOOK_HOST = os.getenv("RENDER_EXTERNAL_URL")  # Render avtomatik beradi
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# 🔹 Logging
logging.basicConfig(level=logging.INFO)

# 🔹 Aiogram sozlamalari
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

# 🔹 Model va cache
user_models = {}
user_cache = {}

print("Modellarni yuklash... Iltimos kuting.")

# 🔹 Modellarni yuklash
blender_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-90M", cache_dir="models/blenderbot")
blender_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-90M", cache_dir="models/blenderbot")

dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir="models/DialogGPT")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", cache_dir="models/DialogGPT")

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
    user_cache[callback.from_user.id] = {}
    await callback.answer(text=f"Model tanlandi: {callback.data.split('_')[1]}")
    await callback.message.reply("Endi xabaringizni yozing, AI javob beradi...")

# 🔹 Xabarlarni qayta ishlash
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

    user_cache[user_id][text] = answer
    await message.reply(answer)

# 🔹 Webhook server (Render uchun)
async def on_startup(app):
    await bot.set_webhook(WEBHOOK_URL)
    print(f"Webhook o‘rnatildi: {WEBHOOK_URL}")

async def on_shutdown(app):
    await bot.delete_webhook()
    await bot.session.close()

async def handle_webhook(request):
    data = await request.json()
    update = types.Update(**data)
    await dp.feed_update(bot, update)
    return web.Response()

def main():
    app = web.Application()
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

if __name__ == "__main__":
    print("🚀 Webhook bot ishga tushdi...")
    main()
