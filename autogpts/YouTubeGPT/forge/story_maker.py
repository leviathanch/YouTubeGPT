from langchain.chains import SequentialChain

class StoryMaker:
    def __init__(self):
        # The sequential chain for story telling
        overall_chain = SequentialChain(
            #chains=[chain1, chain2, chain3, chain4],
            chains=[story_chain1],
            input_variables=["input", "perfect_factors"],
            output_variables=["ranked_solutions"],
            verbose=True
        )
