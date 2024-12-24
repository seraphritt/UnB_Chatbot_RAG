import networkx as nx
from pyvis.network import Network


G = nx.read_graphml('/home/isaquesantos/tg1-isaque/chatbot_git/tcc/graphrag/graph_chunk_entity_relation.graphml')
net = Network()
net.from_nx(G)
net.write_html("knowledge_graph.html")