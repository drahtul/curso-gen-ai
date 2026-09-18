def print_graph(graph, title="Graph structure"):
    print(f"\n{title}:\n")
    try:
        print(graph.get_graph().draw_ascii())
    except ImportError:
        # draw_ascii needs `pip install grandalf`; mermaid always works.
        print(graph.get_graph().draw_mermaid())
