import asyncio
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
import oasis
from oasis import ActionType, AgentGraph, SocialAgent, UserInfo, ManualAction

# Import the custom surrogate wrapper
from surrogate_wrapper import SurrogateRecSys

# Override the OASIS recommendation module
try:
    import oasis.social_platform.recsys as oasis_recsys
except ModuleNotFoundError:
    import oasis.environment as oasis_recsys

async def main():
    # 1. Setup API key and clean previous DB
    os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-testing"
    db_path = "./test_sim.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("1. Loading and injecting TwHIN-BERT surrogate model...")
    my_surrogate = SurrogateRecSys(model_dir="surrogate_model")
    oasis_recsys.twhin_model = my_surrogate
    oasis_recsys.twhin_tokenizer = my_surrogate.model.tokenizer

    print("2. Configuring agents and AgentGraph...")
    agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    agent_graph = AgentGraph()

    for i in range(2):
        agent = SocialAgent(
            agent_id=i,
            user_info=UserInfo(
                user_name=f"user_{i}",
                name=f"User {i}",
                description=f"Test agent {i}",
                profile=None,
                recsys_type="twhin"
            ),
            agent_graph=agent_graph,
            model=agent_model,
            available_actions=[ActionType.LIKE_POST, ActionType.CREATE_POST],
        )
        agent_graph.add_agent(agent)

    print("3. Initializing environment...")
    platform_type = getattr(oasis, "DefaultPlatformType", None)
    plat_val = platform_type.TWITTER if platform_type else "twitter"

    env = oasis.make(
        agent_graph=agent_graph,
        platform=plat_val,
        database_path=db_path
    )
    
    await env.reset()
    
    print("4. Seeding database with ManualAction (no LLM calls)...")
    seed_actions = {
        env.agent_graph.get_agent(i): ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": f"Seed post from agent {i}"}
        )
        for i in range(2)
    }
    await env.step(seed_actions)

    print("5. Triggering recsys refresh with empty action dict...")
    await env.step({})

    await env.close()
    print("\nSUCCESS! The surrogate was successfully queried.")

if __name__ == "__main__":
    asyncio.run(main())
