from livekit.agents import cli

from voice_agent.agent_server import server


if __name__ == "__main__":
    cli.run_app(server)
