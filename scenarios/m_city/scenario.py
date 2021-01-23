import random
from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario


NUM_TRAFFIC_FLOWS = 60

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=60 * 5,
            actors={
                t.TrafficActor(
                    name="car",
                    vehicle_type=random.choice(
                        ["passenger", "bus", "coach", "truck"]
                    ),
                    speed=t.Distribution(1, 0.3),
                    max_speed=10,
                ): 1
            },
        )
        for _ in range(NUM_TRAFFIC_FLOWS)
    ]
)

lane_actor_list = [
    t.SocialAgentActor(name=f"keep-lane-agent-{idx}", agent_locator="zoo.policies:keep-lane-agent-v0",
                       policy_kwargs={'speed': random.choice([5.5, 6, 6.5, 7, 7.5, 8, 9])})
    for idx in range(10)
]

mission_list = [t.Mission(route=t.RandomRoute()) for _ in range(10)]

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_missions={
            "all": (lane_actor_list, mission_list)
        },
    ),
    output_dir=Path(__file__).parent,
)
