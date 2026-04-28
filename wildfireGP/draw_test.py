import matplotlib.pyplot as plt

from wildfireGP.network import NodeState, create_grid
from wildfireGP.render import draw

graph = create_grid(50, 50, water_fraction=0.08, rock_fraction=0.08, seed=42)

# Manually set a few nodes to non-UNBURNED states so all colours are visible.
nodes = list(graph.nodes)
for n in nodes[100:115]:
    graph.nodes[n]["state"] = NodeState.BURNING
for n in nodes[115:130]:
    graph.nodes[n]["state"] = NodeState.BURNED
for n in nodes[130:145]:
    graph.nodes[n]["state"] = NodeState.TREATED

fig, ax = plt.subplots(figsize=(8, 8))
draw(graph, ax=ax)
ax.set_title("draw() manual test")
plt.tight_layout()
plt.savefig("draw_test.png", dpi=150)
print("saved draw_test.png")
