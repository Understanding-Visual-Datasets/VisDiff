UNCONDITIONAL_DIFF_PROMPT = """
    The following are the result of captioning two groups of images:
    {text}

     I am a machine learning researcher and data scientist trying to build an image classifier. Group A are the captions of the image about the bird class in the training set, and Group B are the test set. I want to figure out what kind of distribution shift there are.

    Please write a list of hypotheses (separated by bullet points "*") of how images from Group A differ from those from Group B via their captions. Each hypothesis should be formatted as "Group A ... and Group B...". For example,
    * "Group A is cars in outdoor environments and Group B is cars in indoor environments.”
    * "Group A is dogs with golden hair and Group B is cats with black hair."
    * "Group A is animals walking around at night and Group B is animals drinking water during the day."
    * "Group A and Group B are similar."

    The answers should be pertaining to the content of the images, not the structure of the captions e.g. "Group A has longer captions and Group B has shorter captions" or "Group A is more detailed and Group B is more general" are incorrect answers.
    
    Again, I want to figure out the main differences between these two groups of images. List properties that holds more often for the images (not captions) in group A compared to group B and vice versa. Your response:
    * "Group A
    """

CAPTION_DIFF_PROMPT = """
    The following are the result of captioning two groups of images:
    {text}

    I am a machine learning researcher and data scientist trying to build an image classifier. Group A are the captions of the image about the bird class in the training set, and Group B are the test set. I want to figure out what kind of distribution shift there are.

    Please write a list of 10 hypotheses (separated by bullet points "*") of how images from Group A differ from those from Group B via their captions. Each hypothesis should be formatted as a tuple of captions, the first aligns more with Group A than not Group B and the second aligns more with Group B and not Group A. For example,
    * ("a photo of a car", "a photo of a car in the snow")
    * ("a photo of a dig", "a photo of a dog with black hair")
    * ("animals walking around", "animals drinking water during the day")
    * ("a photo of an object", "a drawing of an object")
    
    Again, I want to figure out the main differences between these two groups of images. List properties that holds more often for the images in group A compared to group B and vice versa. Your response:
    """

VQA_DIFF_PROMPT = """
    The following are the result of asking a VQA model {question} about the following two groups of captions:

    {text}

    Please write a list of hypotheses (separated by bullet points "-") of how images from
    Group A differ from those from Group B via their captions. Each hypothesis should be formatted as "Group A ... and Group B..." and should be with respect to the caption question. Here are three examples:
    - "Group A is cars in mostly outdoor environments and Group B is cars mostly indoor environments.”
    - "Group A is dogs with golden hair and Group B is cats with black hair."
    - "Group A is various animals walking around at night and Group B is various animals drinking water around during the day."
    - "Group A and Group B are similar."

    The answers should be pertaining to the content of the images, not the structure of the captions. Here are examples of incorrect answers:
    - "Group A has longer captions and Group B has shorter captions"
    - "Group A is more detailed and Group B is more general"
    
    Based on the two caption groups (A and B) from the above...
    """

RUIQI_DIFF_PROMPT = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to build an image classifier. Group A are the captions of the image about a class in the training set, and Group B are the test set. I want to figure out what kind of distribution shift are there.

    Come up with 10 short and distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "dog with a long tail"
    * "sunny"
    * "graphic art"
    * "bird with a brown beak"
    * "blurry"
    * "DSLR photo"
    * "person"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Also do not list more than one concept. Here are examples of bad outputs and their corrections:
    * incorrect: "various nature environments like lakes, forests, and mountains" corrected: "nature"
    * incorrect: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * incorrect: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * incorrect: "people in various settings" corrected: "people"
    * incorrect: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * incorrect: "Images containing wooden elements" corrected: "wooden"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images (not captions) in group A compared to group B, with each property being under 5 words. Your response:
"""


RUIQI_DIFF_PROMPT_LONGER_VICUNA = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Also do not list more than one concept. Here are examples of bad outputs and their corrections:
    * incorrect: "various nature environments like lakes, forests, and mountains" corrected: "nature"
    * incorrect: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * incorrect: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * incorrect: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * incorrect: "Images involving interaction between humans and animals" corrected: "interaction between humans and animals"
    * incorrect: "More realistic images" corrected: "realistic images"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*").
    OUTPUT:
"""

RUIQI_DIFF_PROMPT_LONGER = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Also do not list more than one concept. Here are examples of bad outputs and their corrections:
    * incorrect: "various nature environments like lakes, forests, and mountains" corrected: "nature"
    * incorrect: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * incorrect: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * incorrect: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * incorrect: "Images involving interaction between humans and animals" corrected: "interaction between humans and animals"
    * incorrect: "More realistic images" corrected: "realistic images"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""

CLIP_FRIENDLY = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images" 
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""

CLIP_FRIENDLY_GROUP_A = """
    The following are the result of captioning a group of images:

    {text}

    I am a machine learning researcher trying to figure out the major commonalities within this group so I can better understand my data.

    Come up with 10 distinct concepts that appear often in the group. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "presence of ...", "images with ..." are incorrect. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vaccuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"

    Again, I want to figure out the common concepts in a group of images. List properties that hold most often for images (not captions) in the group. Answer with a list (separated by bullet points "*"). Your response:
"""

RUIQI_DIFF_PROMPT_MINIMAL_CONTEXT = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Here are examples of bad outputs and their corrections:
    * bad output: "various nature environments like lakes, forests, and mountains" corrected: "nature environments"
    * bad output: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * bad output: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * bad output: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * bad output: "Images involving interaction between humans and animals" corrected: "interaction between humans and animals"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images in group A compared to group B. Your response:
"""

DIFFUSION_LLM_PROMPT = """
    The following are the result of captioning two groups of images generated by two different image generation models, with each pair of captions corresponding to the same generation prompt:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can correctly identify which model generated which image for unseen prompts.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "dogs with brown hair"
    * "a cluttered scene"
    * "low quality"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption that can be fed into CLIP, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibiliites within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vaccuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images" 
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what the main differences are between these two image generation models so I can correctly identify which model generated which image. List properties that hold more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""