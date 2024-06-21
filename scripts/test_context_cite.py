
# import some packages
import numpy as np
import pandas as pd
import torch
import gc
from context_cite import ContextCiter
from huggingface_hub import login
from transformers import QuantoConfig

# this is the language model we will be using
#model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # This is a chat model
#model_name_or_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" # this is a generic text model
model_name_or_path = "meta-llama/Meta-Llama-3-8B" # this model has a longer context set

# this is the 'background' text that we will attribute importance to
context_text = """
This is Los Angeles. And it's the height of summer.
In a small bungalow off of La Cienega, Clara serves homemade chili and chips in red plastic bowls -- wine in blue plastic.
"The colors don't match and the cups are too big to serve wine.
"You didn't get half the things on my list," she hisses at Gary as he passes.
He shrugs. No one seems to mind but her.
"Did you have to buy a 400 dollar bottle of Scotch?"
Gary waves away her upset. He seems distracted. Guilty of something.
She can tell by the way he avoids her. Clara looks at her face in the hallway mirror.
She's 33 years old. She's slowly getting white hairs in her eyebrows.
She feels completely alone.
Grad students fill the low-ceiling space, chewing big words and gesticulating emphatically.
Gary's face looks up from the photo his parents have framed in expensive mahogany:  Elated, graduating, cap and gown and all.
Toasts and more toasts: "To our Doctor of Philosophy. Amazing Art Prodigy. To Gary Loisin!"
Clara's carefully picked "Happy Graduation" paper banners hang too low from the adobe doors.
People have to duck when they go in the kitchen.

At 74th and Columbus, Apartment 9D, Steven Perdue watches his wife get ready for her violin lesson: black clothes, pearl necklace, her controlled, quiet elegance.
"We cannot ignore this...  We should be vigilant Steven..."
Married 8 years he still can't believe his good luck.
"Steven, are you listening?"
"Yes, yes... He is listening... "
"Sasha broke up with this guy and now I'm missing ten sleeping pills."
"Oh God.  What is his daughter getting into again?"
"We don't want a repeat of last year..."
There is an edge in Margaret's voice: "This girl better grow up..."
She sounds cold and he notices. Margaret pauses by the mirror then touches her neck.
She hates her age. She hates being Sasha's stepmother.
"You think I should talk to her, Maggie?"
Margaret turns and wraps her arms around Steven's stooped frame.
He looks much older than 50. "She doesn't talk to me," Margaret whispers.
"So yes, I think you should do it." She stops. "I'll be home late tonight," she adds.
She wants Steven to ask why, but he doesn't.

The Graduation party keeps going.
Clara stares at her husband, once the most handsome man in the room: Gary's 35 and already balding.
His face is liquor-flushed as he bends over a 4-month-old baby-boy.
Son of friends. His friends, not hers. He asks to hold the boy, and Clara's eyes melt.
In an alternate universe this boy would be theirs.
Gary's awkward, the baby starts crying and everyone laughs: "You're not daddy material, man..."
Clara can see the back of Gary's neck; all red... Embarrassed, but hiding it.
She knows this neck. Has studied it each night for ten years.
And Clara is suddenly filled with the need to protect him...
She puts her hand on the small of his back, hoping he'd turn.
The baby's father steps up.
"So, is it time to open that Scotch?"
"Clara, you're moving to New York too? Going where -- doing what?"
Clara looks at Gary confused and he smiles, past her -- to the crowd.
He's basking in the glory of something.
Tommy, the baby's father, looks at Clara's face and takes a step back -- "Oh man... Did I blow the surprise? I'm so sorry..."

"You guys are into bullshit drama again."
"I just like to sleep deeply these days. That's all."
Sasha strides up to her dad and opens her palm.
The ten white sleeping pills stare at him, an act of defiance.
"I certainly don't need you guys freaking out..."
Sasha watches her dad flush the pills down the toilet.
"You should have put them back in the pill-box."
Of course Sasha's right and now his gesture of closure feels stupid.
He puts his hand on her shoulder but she wiggles away.
"So how do you sleep these days dad?"
He turns on CNN. "Have you told Margaret yet?"
Steven stares at the TV and carnage.
"You haven't told my stepmother you've lost all your money? That your stocks have all tanked? That I might have to drop out of college?"
On the TV -- the Middle East, India, Wars, Earthquakes, Crashes, Steven sighs.
"There's no need to disturb Margaret... I'll figure it out.
"What do you want for your birthday?" he asks.
"I want you to start telling the truth," says Sasha and her voice, for the first time is breaking.
"Your duplicity is so fucking exhausting..."

Clara stares at Jeannie -- she must be what, 25? 26?
So in love with this baby spitting up goo on her lap...
"And then, the moment they found out I was pregnant, Tommy's parents swept in and erased all our debt.
All 10,000 dollars in one check... So we could start fresh and not stress... Amazing. Just amazing. I love them."
"That's great, Jeannie," Clara musters.
"You want to have kids?" asks Jeannie. Still pudgy from the birth. So full of life and good cheer.
"Not yet," Clara says stiffly. "Can't afford it for one."
Jeannie looks up, expecting more of an answer:
"I don't really like modern art, installation art? I don't really get it," admits Jeannie
"Tommy says it's an incredible opportunity for Gary. A chance to create his own, site-specific piece, downtown in New York..."
"We'll see," Clara says.
She hates Gary for making her the killjoy.
She takes out more bottles of wine from the cupboard.
People pat her on her back: "You must be so proud of him Clara."
She doesn't care anymore how much wine she'll serve... At least one bottle per person these days.

Inside the cab, Margaret holds onto her violin case.
"All the way down Columbus, OK?"
She gives the instructions in a pleasant stiff voice and pulls out her cell.
She cancels her violin lesson.
The cab driver's eyes keep checking her out.
She pauses, hits voice mail.
Alexander's accent purrs in her ear. He's waiting for her.
"So what's eating you princess?" asks the cabdriver.
"He's Sikh. His turban blocks her view of the asphalt...
"Whatever it is, tell your husband. Don't keep the stuff in. That's what gives people cancer."
Alexander's name vibrates on her phone.
Margaret texts him instead:  I'm running a few minutes late.
The eyes of the cabdriver are still there in the mirror 
"Do you have children princess?"
She shakes her head, no, without thinking.
And then she's terrified by what she's done. Denying Sasha's existence.
But it's too late to undo it.
"You look like someone who should," says the cabdriver and is quiet the rest of the way.
She pays with a twenty, as they pull by the fountain, and hates herself, because -- as if to prove him right -- she overtips.
"""

# this is the 'query' text that we will use to assign attributions
query_text = """
"No art residencies for us," Jeannie laughs and nods at the gurgling boy on her lap.
Tommy struts in, dangling his scotch by his crotch.
He holds a cracker, piled with over-ripe brie and jelly, shoves it in his wife's mouth.
She thankfully chews, like a chipmunk.
His black faded t-shirt walks towards the kitchen.
"I'm an existentialist-fatalist," it boasts.
Then he ducks. The "Congratulations!" sign still hits him on the back.
"One more Tommy, please!"
She's never losing the baby weight that way, Clara thinks.
But who the fuck cares... she's young...  Clara looks at the baby who's drooling over his hands and walks out of the room.
"How could you do this to me?"" Clara hisses at Gary.
"When did you find out about New York? When did you even apply?"
Anger and humiliation are drowning her breath.
And probably a touch too much wine.
But Gary is giddy and thrilled and sees nothing wrong with their lives.
"Baby -- be proud," he laughs.
"It's just for a year, we'll work it out..."
And what is so sad -- he really believes this...
"""

# decide the level at which we compute attributions ('paragraph', 'sentence', or 'word')
source_type = "paragraph"

# probability of 'dropping out' each block when evaluating their contribution (default 0.5)
ablation_keep_prob = 0.5

# they fit an L1 regularized model to estimate the importance of scenes, and this is the regularization strength
solver_alpha = 1e-3

model_kwargs = {}
if "Meta" in model_name_or_path:
    # need to quantize to float8 to fit on a single GPU
    model_kwargs["quantization_config"] = QuantoConfig(weights="float8")
    
    # need to pass authentication key if this hasn't been configured already
    login()
    
    # maximum length of the context text (in units of tokens)
    max_context_length = 8000
elif "tiny" in model_name_or_path: # using tinyllama
    # maximum length of the context text (in units of tokens)
    max_context_length = 2048
else:
    raise NotImplementedError
    
# instantiate model
cc = ContextCiter.from_pretrained(model_name_or_path, context_text, solver_alpha = solver_alpha, source_type = source_type, model_kwargs=model_kwargs)

# set the query for the model
cc.set_query(query_text)

# print the full output (context + query) just to check that things vaguely work
print(cc._output) 
# make sure that we're not exceeding the maximum context length of the model
assert len(cc._output_tokens["input_ids"]) <= max_context_length 

# delete cached logit probs in case we have changed something (otherwise we will use the cached results)
cc._cache["reg_logit_probs"] = None
# compute 'attributions'
results = cc.get_attributions(as_dataframe=True, top_k=10)
print(results) # print stuff

