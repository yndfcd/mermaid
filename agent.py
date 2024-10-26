from __future__ import annotations

import logging
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

instructions = (
            "You are a restaurant order assistant powered by a large language model. You will:"
            "1. Respond only to menu-related requests, reservations, and order modifications."
            "2. If the customer provides irrelevant or unrelated input, you will politely redirect them to valid options."
            "3. Keep track of all ordered items and modifications. Context must always be maintained throughout the conversation."
            "4. You must use function calls to retrieve menu and availability data from the backend."
            "5. When summarizing the order, any items outside the menu must be excluded automatically."
            "6. For example:"
                "- If the customer orders something that is not on the menu, respond with: \"That item is not available.\""
                "- If the customer modifies an order, such as changing a side dish, make the appropriate adjustment."
            "7. If an item is unavailable, let the customer know and suggest an alternative from the menu."
        )


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    logger.info("starting multimodal agent")

    print(instructions)
    
    model = openai.realtime.RealtimeModel(
        instructions=instructions,
        modalities=["audio", "text"],
    )
    assistant = MultimodalAgent(model=model)
    assistant.start(ctx.room, participant)

    session = model.sessions[0]
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content="Please begin the interaction with the user in a manner consistent with your instructions.",
        )
    )
    session.response.create()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )

