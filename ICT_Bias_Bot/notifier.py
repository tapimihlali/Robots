import requests
import config

def escape_markdown_v2(text):
    """Escapes characters that have special meaning in MarkdownV2."""
    # List of characters that need to be escaped in MarkdownV2
    # See https://core.telegram.org/bots/api#markdownv2-style
    escape_chars = '_*[]()~`>#+-=|{}.!\\'
    escaped_text = ""
    for char in str(text):
        if char in escape_chars:
            escaped_text += '\\' + char
        else:
            escaped_text += char
    return escaped_text

class Notifier:
    """
    Handles sending notifications via Telegram using requests library.
    """
    def __init__(self):
        if config.TELEGRAM_ENABLED:
            self.telegram_bot_token = config.TELEGRAM_BOT_TOKEN
            self.telegram_chat_id = config.TELEGRAM_CHAT_ID
            print(f"DEBUG: Notifier init - Token: {self.telegram_bot_token[:5]}... (masked), Chat ID: {self.telegram_chat_id}")
            if not self.telegram_bot_token or not self.telegram_chat_id:
                print("Telegram credentials not fully set in config. Notifications disabled.")
                self.enabled = False
            else:
                self.enabled = True
                print("Telegram Notifier initialized using requests.")
        else:
            self.enabled = False
            print("Telegram Notifier is disabled in the config.")

    def send_message(self, message):
        """Sends a message to the configured Telegram chat."""
        if not self.enabled:
            print("DEBUG: Notifier disabled, message not sent.")
            return
        
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'MarkdownV2'
        }
        print(f"DEBUG: Sending message to URL: {url}")
        print(f"DEBUG: Sending notification.")
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"Successfully sent Telegram message. Response: {response.text}")
            else:
                print(f"Failed to send Telegram message. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"An exception occurred while sending Telegram message: {e}")