
import re
from playwright.sync_api import Page, expect
import pytest

def test_ai_response(page: Page):
    """
    Test that the AI returns a response in the UI.
    """
    
    # Capture console messages
    page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))

    page.goto("http://localhost:3000/")

    # Wait for the chat input to be visible
    input_selector = "textarea[placeholder='Ask a question about your documents...']"
    page.wait_for_selector(input_selector, timeout=60000)

    # Type a message into the chat input
    page.locator(input_selector).fill("Hello, world!")

    # Click the send button
    page.get_by_role("button", name="➤").click()

    # Wait for the assistant's response
    assistant_response = page.locator("div[class*='ChatMessage_message__'][role='assistant']")

    try:
        # Wait for the response to be visible with a longer timeout
        expect(assistant_response).to_be_visible(timeout=30000)
        expect(assistant_response).not_to_be_empty()
        expect(assistant_response).not_to_contain_text("Error:")
    except Exception as e:
        # If the response is not found, print the messages list for debugging
        print(page.locator("div[class*='ChatInterface_messagesList__']").inner_html())
        raise e
