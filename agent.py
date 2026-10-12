"""
A minimal AI agent built from scratch against the Anthropic API, with no
agent framework involved.

Contains:
- A tool (get_order_status) with a schema the model can read
- An Agent class that runs the tool-calling loop
- Memory: self.messages persists across calls to Agent.run()

Companion code for "How and Why to Build an AI Agent From Scratch in Python."
"""
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = (
    "You are a helpful assistant who can answer order status questions "
    "using built-in tools available to you. Do not answer any other questions "
    "not related to order."
)

# ---------------------------------------------------------------------------
# Tool: order lookup
# ---------------------------------------------------------------------------

orders_db = {
    "4471": {"status": "shipped", "carrier": "UPS", "eta": "2 days"},
    "4472": {"status": "processing", "carrier": None, "eta": None},
}


def get_order_status(order_id: str) -> dict:
    return orders_db.get(order_id, {"error": "No order found with that ID"})


get_order_status_schema = {
    "name": "get_order_status",
    "description": (
        "Looks up the current status of a customer order by its ID. "
        "Use this any time a question depends on current order data "
        "rather than general policy information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID to look up, e.g. '4471'",
            },
        },
        "required": ["order_id"],
    },
}

TOOLS = [get_order_status_schema]
TOOL_MAP = {"get_order_status": get_order_status}


# ---------------------------------------------------------------------------
# Agent with memory & streaming
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, tools, tool_map, max_iterations: int = 6):
        self.tools = tools
        self.tool_map = tool_map
        self.max_iterations = max_iterations
        self.messages = []

    def run(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            has_printed_header = False

            with client.messages.stream(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.messages,
            ) as stream:
                for text_delta in stream.text_stream:
                    # Print "Agent: " header only when the first actual text token arrives
                    if not has_printed_header:
                        print("Agent: ", end="", flush=True)
                        has_printed_header = True

                    print(text_delta, end="", flush=True)

                final_message = stream.get_final_message()

            # Add a newline only if assistant text was outputted
            if has_printed_header:
                print()

            self.messages.append({"role": "assistant", "content": final_message.content})

            if final_message.stop_reason != "tool_use":
                return

            tool_results = []
            for block in final_message.content:
                if block.type == "tool_use":
                    function = self.tool_map[block.name]
                    output = function(**block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output),
                        }
                    )
            self.messages.append({"role": "user", "content": tool_results})

        print("Stopped after max iterations without a final answer.")


if __name__ == "__main__":
    agent = Agent(tools=TOOLS, tool_map=TOOL_MAP)

    print("How can I help you today? (Type 'exit' or 'quit' to end)")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            agent.run(user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break