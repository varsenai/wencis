from __future__ import annotations

import asyncio
import secrets
import httpx
from typing import Any


class TelegramPairingSession:
    """
    Handles the 'Magic Handshake' pairing flow for Telegram bots.
    """

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.pairing_secret = f"PAIR-{secrets.token_hex(3).upper()}"
        self.bot_username = ""

    async def get_bot_info(self) -> dict[str, Any]:
        """Fetch bot info to get the username for the deep link."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/getMe")
            data = resp.json()
            if not data.get("ok"):
                raise ValueError(f"Invalid Bot Token: {data.get('description')}")
            self.bot_username = data["result"]["username"]
            return data["result"]

    def get_deep_link(self) -> str:
        """Generate the t.me deep link with the pairing secret."""
        if not self.bot_username:
            raise RuntimeError("Bot info not fetched. Call get_bot_info() first.")
        return f"https://t.me/{self.bot_username}?start={self.pairing_secret}"

    async def wait_for_handshake(self, timeout_s: int = 120) -> int:
        """
        Poll for the pairing secret in the bot's incoming messages.
        Returns the chat_id once verified.
        """
        async with httpx.AsyncClient() as client:
            offset = 0
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < timeout_s:
                try:
                    resp = await client.get(
                        f"{self.base_url}/getUpdates",
                        params={"offset": offset, "timeout": 10},
                    )
                    data = resp.json()
                    if not data.get("ok"):
                        continue

                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        message = update.get("message", {})
                        text = message.get("text", "")
                        
                        # Check if the message contains our secret
                        if self.pairing_secret in text:
                            chat_id = message["chat"]["id"]
                            # Send the final greeting
                            await self.send_greeting(chat_id)
                            return chat_id
                            
                except Exception:
                    pass
                
                await asyncio.sleep(1)
            
            raise TimeoutError("Pairing timed out. Please try again.")

    async def send_greeting(self, chat_id: int):
        """Send the first message to the user once paired."""
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "🛡️ **VYN IDENTITY VERIFIED**\n\nYour cognitive link is now active. I am ready to assist you.",
                    "parse_mode": "Markdown",
                },
            )
