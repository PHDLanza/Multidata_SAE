
GUIDELINES_VISUAL_GENERATION= """ 
            [REQUIREMENTS]
                Focus only on the highlighted region in each image. If no region is highlighted or if the highlighted region is minimal (e.g., a few bright spots), ignore the image.
            
                Identify common visual patterns, objects, or concepts in the activated regions. For example, note if highlighted areas show consistent structures, such as mesh patterns or similar objects.\
            
            [GUIDELINES]

            1.Consider Text Context: While maintaining primary focus on the highlighted regions in images, you may marginally consider the associated text (questions and answers) to support or refine your visual observations. 
            However, the final concept should be predominantly based on visual patterns.
            
            2.Concise Description Only: Provide a short, direct description of the common features within the highlighted regions. Avoid any interpretive language—simply state what you see, such as “mesh-like structures” or “actions related to joy or happiness”
            
            3. Describe Only the Highlighted Regions: Generate captions solely based on the highlighted regions. If no meaningful pattern is visible, or if only a few scattered spots are highlighted,
                output: \"Concept:  `No visual concept`\ 
            """

GUIDELINES_TEXTUAL_GENERATION=""" 
        [REQUIREMENTS]
               Focus only on the text content provided with each example. If the text is missing, irrelevant, or extremely minimal (e.g., a few unrelated words), ignore the text.

            Identify common themes, objects, or concepts mentioned across the text snippets. Pay special attention to any highlighted word in each text—this word should be treated as the most important cue for concept identification.

        [GUIDELINES]

            1.Consider Visual Context: While maintaining primary focus on the text, with words in parentheses as priority,  you may marginally consider the associated image to support or refine your textual observations. 
                However, the final concept should be predominantly based on textual patterns.


            2.Concise Description Only: Provide a short, direct description of the common concept emerging from the texts. Avoid speculation or abstract interpretation—simply state what is explicitly or implicitly repeated, especially in relation to the highlighted words (e.g., "vehicles," "cooking actions," "types of animals").
            

            3.If no clear concept emerges from the texts (e.g., if they are too diverse or vague), write:  \"Concept:  `No textual concept`\"

  
            """
GUIDELINES_LABELING=""" 
[GUIDELINES] You are an AI assistant tasked with assigning a single label based on the given input text. Each input will contain a description ,
    → of a visual feature, which you must categorize into one of the following classes:  
    scene − Describes a scene or environment. 
    object − Describes an object or entity. 
    part − Describes a part or aspect of an object.
    material − Describes a material or substance that constitutes other objects. 
    texture − Describes the texture of an object. 
    color − Describes the color of an object.  
    Please provide only the class label from the list [scene, object, part, material, texture, color] with no additional text. 
    Only one ,→ label should be chosen. Make sure you only choose from the classes listed above and do not output any other classes.  

    Categorize the following description:  {description}  

    ANSWER:
"""
FSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain latent of text, such as "male pronouns" or "text with negative sentiment". You will be given a few examples of text that contain this latent. Portions of the sentence which strongly represent this latent are between tokens << and >>.

Some examples might be mislabeled. Your task is to determine if every single token within << and >> is correctly labeled. Consider that all provided examples could be correct, none of the examples could be correct, or a mix. An example is only correct if every marked token is representative of the latent

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
FSCORER_EXAMPLE_ONE = """Latent explanation: Words related to American football positions, specifically the tight end position.

Test examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots<< tight end>> Rob Gronkowski had his bossâĢĻ
Example 1: posted<|endoftext|>You should know this<< about>> offensive line coaches: they are large, demanding<< men>>
Example 2: Media Day 2015ĊĊLSU<< defensive>> end Isaiah Washington (94) speaks<< to the>>
Example 3:<< running backs>>," he said. .. Defensive<< end>> Carroll Phillips is improving and his injury is
Example 4:<< line>>, with the left side âĢĶ namely<< tackle>> Byron Bell at<< tackle>> and<< guard>> Amini
"""

# DSCORER_RESPONSE_ONE = """{
#   "example_0": 1,
#   "example_1": 0,
#   "example_2": 0,
#   "example_3": 1,
#   "example_4": 1
# }"""

FSCORER_RESPONSE_ONE = "[1,0,0,1,1]"

# https://www.neuronpedia.org/gpt2-small/6-res-jb/9396
FSCORER_EXAMPLE_TWO = """Latent explanation: The word "guys" in the phrase "you guys".

Test examples:

Example 0: if you are<< comfortable>> with it. You<< guys>> support me in many other ways already and
Example 1: birth control access<|endoftext|> but I assure you<< women>> in Kentucky aren't laughing as they struggle
Example 2:âĢĻs gig! I hope you guys<< LOVE>> her, and<< please>> be nice,
Example 3:American, told<< Hannity>> that âĢľyou<< guys>> are playing the race card.âĢĿ
Example 4:<< the>><|endoftext|>ľI want to<< remind>> you all that 10 days ago (director Massimil
"""

# DSCORER_RESPONSE_TWO = """{
#   "example_0": 0,
#   "example_1": 0,
#   "example_2": 0,
#   "example_3": 0,
#   "example_4": 0
# }"""

FSCORER_RESPONSE_TWO = "[0,0,0,0,0]"

# https://www.neuronpedia.org/gpt2-small/8-res-jb/12654
FSCORER_EXAMPLE_THREE = """Latent explanation: "of" before words that start with a capital letter.

Test examples:

Example 0: climate, TomblinâĢĻs Chief<< of>> Staff Charlie Lorensen said.Ċ
Example 1: no wonderworking relics, no true Body and Blood<< of>> Christ, no true Baptism
Example 2:ĊĊDeborah Sathe, Head<< of>> Talent Development and Production at Film London,
Example 3:ĊĊIt has been devised by Director<< of>> Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director<< of>> Athletics? Â· Finally, we believe the
"""

# DSCORER_RESPONSE_THREE = """{
#   "example_0": 1,
#   "example_1": 1,
#   "example_2": 1,
#   "example_3": 1,
#   "example_4": 1
# }"""

FSCORER_RESPONSE_THREE = "[1,1,1,1,1]"
DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain latent of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Your task is to determine which examples possess the latent.

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
DSCORER_EXAMPLE_ONE = """Latent explanation: Words related to American football positions, specifically the tight end position.

Test examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ
Example 1: names of months used in The Lord of the Rings:ĊĊâĢľâĢ¦the
Example 2: Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the
Example 3: shown, is generally not eligible for ads. For example, videos about recent tragedies,
Example 4: line, with the left side âĢĶ namely tackle Byron Bell at tackle and guard Amini
"""

DSCORER_RESPONSE_ONE = "[1,0,0,0,1]"

# https://www.neuronpedia.org/gpt2-small/6-res-jb/9396
DSCORER_EXAMPLE_TWO = """Latent explanation: The word "guys" in the phrase "you guys".

Test examples:

Example 0: enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both
Example 1: birth control access<|endoftext|> but I assure you women in Kentucky aren't laughing as they struggle
Example 2: du Soleil Fall Protection Program with construction requirements that do not apply to theater settings because
Example 3:Ċ<|endoftext|> distasteful. Amidst the slime lurk bits of Schadenfre
Example 4: the<|endoftext|>ľI want to remind you all that 10 days ago (director Massimil
"""

DSCORER_RESPONSE_TWO = "[0,0,0,0,0]"

# https://www.neuronpedia.org/gpt2-small/8-res-jb/12654
DSCORER_EXAMPLE_THREE = """Latent explanation: "of" before words that start with a capital letter.

Test examples:

Example 0: climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ
Example 1: no wonderworking relics, no true Body and Blood of Christ, no true Baptism
Example 2:ĊĊDeborah Sathe, Head of Talent Development and Production at Film London,
Example 3:ĊĊIt has been devised by Director of Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director of Athletics? Â· Finally, we believe the
"""

DSCORER_RESPONSE_THREE = "[1,1,1,1,1]"

DGENERATION_PROMPT = """Latent explanation: {explanation}

Text examples:

{examples}
"""



# GUIDELINES_VISUAL_GENERATION_OLD= """ 
#             [REQUIREMENTS]

#             1. Focus only on the highlighted region in each image. If no region is highlighted or if the highlighted region is minimal (e.g., a few bright spots), ignore the image.
#             2. Identify common visual patterns, objects, or concepts in the activated regions. For example, note if highlighted areas show consistent structures, such as mesh patterns or similar objects.
            
#             [GUIDELINES]
            
#             1.You will receive a series of images and correlated texts, and you have to identify the shared concept between them.he images will be masked, so you will have to describe only on the visible portion of image to generate a concept.
#             These are samples taken from a Visual Question Answering dataset, so for each image there is a question and an answer.
            
            
#             2. Concise Description Only: Provide a short, direct description of the common features within the highlighted regions. Avoid any interpretive language—simply state what you see, such as “mesh-like structures” or “actions related to joy or happiness”. 
#             Concepts can be only visual concepts so related to the image, the text can be only used to guide the generation, such as if you see a series of images regarding a specific race of dog look also if the all texts, or part of them, mention the race dog.

            
#             3. If no clear concept emerges from the images to any visual concept, for example if the pixels are too far sparse that cannot form any understandable figure, write: No visual concept 

#             [OUTPUT EXAMPLES]
#             - Concept: "A tennis racket"   
        
#             - Concept: "No visual concept"   
            
            
            
#             Remember,Write always only one Concept for the entire set of inputs
#         """
# GUIDELINES_TEXTUAL_GENERATION_OLD=""" 
#             [REQUIREMENTS]

#                 Focus only on the text content provided with each example. If the text is missing, irrelevant, or extremely minimal (e.g., a few unrelated words), ignore that example.

#                 Identify common themes, objects, or concepts mentioned across the text snippets. Pay special attention to any highlighted word in each text—this word should be treated as the most important cue for concept identification.

#             [GUIDELINES]

#                 1.You will receive a series of text snippets, sometimes accompanied by images. Only use the text, and in particular the word between parentheses, to identify the shared concept. Images should not be considered in your analysis.
#                 These examples are derived from a Visual Question Answering dataset, so each text is in the form of a question or an answer.

#                 2.Concise Description Only: Provide a short, direct description of the common concept emerging from the texts. Avoid speculation or abstract interpretation—simply state what is explicitly or implicitly repeated, especially in relation to the highlighted words (e.g., “vehicles,” “cooking actions,” “types of animals”).
#                 Use the image only for reference if absolutely necessary; the main analysis must be text-driven, with words in parentheses as priority.

#                 3.If no clear concept emerges from the texts (e.g., if they are too diverse or vague), write: No textual concept

#             [OUTPUT EXAMPLES]

#                 Concept: "A tennis match"

#                 Concept: "Descriptions of birds"

#                 Concept: "No textual concept"
                
#             Remember,Write always only one Concept for the entire set of inputs
#             """