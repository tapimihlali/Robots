# -*- coding: utf-8 -*-
"""
Notifier for sending messages.
"""

import requests
import config

def escape_markdown_v2(text):
    """Escapes text for Telegram's MarkdownV2 format."""
    escape_chars = '_*[]()~`>#+-=|{}.!'
    return "".join(['\\' + char if char in escape_chars else char for char in text])

class Notifier:
    def __init__(self):
        self.enabled = config.TELEGRAM_ENABLED
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, message):
        """Sends a message using Telegram or prints to console."""
        if self.enabled and self.token and self.chat_id:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'MarkdownV2'
            }
            try:
                response = requests.post(self.url, json=payload)
                if response.status_code != 200:
                    print(f"Error sending Telegram message: {response.text}")
            except Exception as e:
                print(f"Exception sending Telegram message: {e}")
        else:
            print(f"NOTIFICATION: {message}")
