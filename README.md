# [B.E.S.T.](https://www.youtube.com/watch?app=desktop&v=XTte01kdG_k&ab_channel=NeilCicieregaMusic)
# (Acronym TBD...  Something Something Search Tree?)

A mashup of Eureka and PokemonRedExperiments.  Can an AI on a self-improvement reward-function-rewriting loop become the very best, like no one ever was?

**USE WITH CAUTION - THIS LETS AN AI RUN SOMEWHAT-ARBITRARY CODE ON YOUR COMPUTER**

# About Fork

This is an early experiment by a longtime-coder learning RL this year, wanting to combine some of the more mindblowing yet simple examples of ML learning progress, on a hunt for lowhanging fruit self-improvement loops!

Meant to be combined as [BEST-Eureka](https://github.com/dogcomplex/BEST-Eureka) fork of NVidia's Eureka code and [BEST-Pokemon](https://github.com/dogcomplex/BEST) fork of PokemonRedExperiments repos.  Major thanks to original works!

Our goal is to see if GPT4 (or local LLMs) can use the Eureka-proven self-improvement loop for more general tasks like playing Pokemon.  There are probably simpler gyms to try this on, but few as fun as this!  Feels achievable!

## Does this work?

Kinda? Maybe? Soon?

Current state (Dec 2023) is Eureka-style code iterating loop with GPT4 (especially lovely 128k context) successfully(?) iterates on the `PokemonRedExperiments/baselines/red_gym_env.py` files to write and test new reward functions automatically, as well as (hopefully?) saves training sessions to the `PokemonRedExperiments/sessions` folder, pulling the latest saved session by default.  This is largely governed by (hacky, low-effort) modifications to `Eureka/eureka/eureka.py`, `PokemonRedExperiments/baselines/red_gym_env.py`, and `PokemonRedExperiments/baselines/run_baseline_parallel_fast.py` files to get baseline functionality.  Progress in this all is governed primarily by me understanding the underlying code better and fixing things that weren't working (like loading sessions instead of starting from scratch each time...).  

Last I tested, sessions generally get to Brock's gym, die to sandshrew, and would keep going fine except they open up the start menu and never escape it!  Probably could be solved by prompting the reward function to disincentivize being in the menu, but curious if it could figure that out naturally.  Have seen previous iterations make it to Mt Moon.

Still unclear without considerably more training whether the Eureka-style looping here is actually improving (or doing!) anything, or if it's all pretraining.  Need to get GPU training back in and let this iterate a while first.  Likely doing something fundamentally wrong too like continually retraining weights on new reward functions that turn it into gibberish...  but hey, if we can't do this all as newbie RL coders mashing things together through brute force then what's the point?  Just gotta get it working enough that much more qualified folks (like GPT4) can iterate it to success!


# TODOS

Overall the whole system could benefit from a rewrite from scratch, honestly.  Eureka's code was very baked into their original physics gym and it probably would have been better to just write the LLM-querying loop and statistics extraction from scratch in a more general way than adapting theirs - but hindsight's 20-20, and it's all part of experimentation!

Likewise, PokemonRed could probably be consolidated better so its reward function code is separated from gym actions, so the LLM can just focus on iterating a single function/set.  It's also super likely there are fundamental limitations to automated reward function tuning with such a complex task - we probably need some sort of specialized hierarchy of subtask tuning and ability to swap (though this might just be a different "mode" if/then conditional logic in the single-reward-function version!  It sure would be neat if the Eureka loop managed to discover that itself and started specializing autonomously, eh? 🙃).  Goal here is to push simple reward iteration as far as it can go first and find out!

## Specific Tasks TODO:

- get this iterating via GPU (again).  See the `PokemonRedExperiments\baselines\ray_exp` folder for inspiration from the original creator!  First I gotta get my shit Windows to recognize CUDA correctly...  but others should have much better luck
- refactoring for clarity... all around.  `eureka.py` was not particularly written for clarity to begin with, and poor `run_baseline_parallel_fast.py` has been bastardized all to hell
- tweak/simplify prompting infrastructure (and text files).  bit of a mess to begin with
- make session recovery smarter - e.g. even just tracking previous best session instead of recovering from latest (lol)
- improve reporting/statistics
- let code writer replace the whole red_gym_env.py code instead of just compute_success() function.  Silly limitation really, just to kludge Eureka in.
- lol wrap this all in some security/containers so there's not an autonomous unsupervised exec loop going on 🙃
- refactor out/split `red_gym_env.py` to be smaller and only hold relevant reward function code.  Smaller that gets the less tokens and better quality of iterations.
- revitalize original functionality - poor original eureka and pokemon runs are both basically broken to make this mutant work
- local LLM integration!
- passing screenshots to prompts via GPT4-vision!  Might help for automated escaping of dumb traps (like start menu looping).  The fact it would have to tweak the reward function and retrain to escape makes this tricky to grok though
- much of the above could probably be done better via a full infrastructure rewrite, but that's just me!


https://github.com/eureka-research/Eureka
https://github.com/PWhiddy/PokemonRedExperiments

## Installation:

(oh god)

1. Follow standard Eureka repo installation [(Old Instructions)](https://github.com/dogcomplex/BEST-Eureka/blob/master/README_old.md).  No real change here from original:

1.a) 
```
conda create -n eureka python=3.8
conda activate eureka

git clone https://github.com/eureka-research/Eureka.git
cd Eureka
```
(3.10 should work too)

1.b) Can attempt this part, but not strictly necessary for Pokemon stuff (and seems to be impossible for plain Windows?)
```
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
(test installation) python examples/joint_monkey.py
cd ../..
```

1.c) Run all this (and install anything it errors on, if you're like me):
```
cd Eureka
pip install -e .
cd isaacgymenvs
pip install -e .
cd ../rl_games
pip install -e .
```

1.d) And of course:
```
export OPENAI_API_KEY= "YOUR_API_KEY"
```

2. Now for the [Pokemon part (Instructions)](https://github.com/dogcomplex/BEST/blob/master/README_old.md).  Only tweak is we want to integrate the requirements.yml into conda env:

2.a) 
```
Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to PokemonRed.gb if it is not already. The sha1 sum should be ea9bcae617fdf159b045185467ae58b2e4a48b9a, which you can verify by running shasum PokemonRed.gb.
```

2.b) From Eureka repo:
```
cd PokemonRedExperiments/baselines
conda env update --file requirements.yml --name eureka
conda activate eureka
```
(note this is probably horrible practice for conda dependency management!)

2.c) This probably breaks - just because of sloppy integrations.  Test out with original Poke repo if you're feeling unconfident though
```
python run_pretrained_interactive.py
```

## Execution

```
cd Eureka/eureka
python eureka.py env=red_gym_env sample=4 iteration=10 model=gpt-3.5-turbo-1106
```
OR
```
cd Eureka/eureka
python eureka.py env=red_gym_env sample=4 iteration=10 model=gpt-4-1106-preview
```
(note, these get expensive!  Like 20 cents per iteration for gpt-4-1106-preview.  3.5 much cheaper and better for testing.  Current setup requires the large-context models but some mild tweaking (mostly shrinking code file and the prompts) could shrink things down.  Will do if/when we get to serious iterating, but likely running a local model by then anyway)



Probably a few more loose packages you'll see you're missing as you try and run it.  `conda install <pkgname>` should do the trick.

`eureka\outputs\eureka` folder will show you logs for each session, along with printouts of what's being sent to OpenAI API and PokemonRed outputs for each code variation.  Good for debugging execution errors in the poke code that aren't just being generated by ChatGPT.   You can see each iteration attempt log output at e.g. `env_iter0_response0.txt`.  If it's showing output like 
```
step:      0 money_reward:  0.00 poke_count_reward:  0.04 xp_reward:  0.28 events_reward:  0.04 badges_reward:  0.00 explore_reward:  0.00 task_score:  1.80 event: 11.00 level:  0.00 explore:  0.00 sum: 13.16
```
congrats - you've got a running game.  Switch `run_baseline_parallel_fast.py` to `'headless': False,` if you want to see live gameplay (with considerable slowdown cost).  Set `num_cpu=1` to more to see multiple instances. (debatable whether it's better to split instances there or give each one a unique reward function and just increase `sample=4` in original calling command.  Seems like we want to ideally give each code iteration a few test runs to account for stochasticity?  Meh - will probably just make that a meta-param and have the Eureka-loop tune on that too!)

`iter1_response0_to_4.txt` etc are printouts of all the back and forths to GPT4.  Useful to see how it's reasoning, and what info it's getting back between iterations.  If the power lies anywhere in this architecture, it'll be in the reward statistics e.g.:

```
money_reward: ['0.00', '0.00', '0.01', '0.01', '0.01', '0.01', '0.01', '0.01', '0.01', '0.01'], Max: 0.01, Mean: 0.01, Min: 0.00 
poke_count_reward: ['0.04', '0.08', '0.09', '0.09', '0.09', '0.09', '0.09', '0.09', '0.09', '0.15'], Max: 0.15, Mean: 0.09, Min: 0.04 
xp_reward: ['0.03', '0.09', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.39'], Max: 0.39, Mean: 0.15, Min: 0.03 
events_reward: ['0.04', '0.05', '0.05', '0.05', '0.05', '0.05', '0.05', '0.05', '0.05', '0.06'], Max: 0.06, Mean: 0.05, Min: 0.04 
badges_reward: ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '2.50'], Max: 2.50, Mean: 0.36, Min: 0.00 
explore_reward: ['0.00', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03'], Max: 0.03, Mean: 0.03, Min: 0.00 
event: ['11.00', '12.00', '12.00', '12.00', '12.00', '12.00', '12.00', '12.00', '13.00', '14.00'], Max: 14.00, Mean: 12.21, Min: 11.00 
level: ['0.00', '6.00', '7.00', '7.00', '7.00', '7.00', '7.00', '7.00', '7.00', '15.00'], Max: 15.00, Mean: 7.93, Min: 0.00 
explore: ['0.00', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03'], Max: 0.03, Mean: 0.03, Min: 0.00 
sum: ['11.12', '18.26', '19.31', '19.31', '19.31', '19.31', '19.31', '19.31', '20.31', '32.17'], Max: 32.17, Mean: 20.85, Min: 11.12 
task_score: ['2.00'], Max: 2.00, Mean: 2.00, Min: 2.00 
```
(if this feels crude or incomplete to you, that's exactly where this all needs more work!  See the `eureka.py` code extracting and processing this)


## Modifying/Coding:

A few gotchas:
 - `PokemonRedExperiments/baselines/red_gym_env.py` is not actually called directly in instances.  I think(?) it's what's passed to GPT to generate code, but then `env_iter0_response0.py` (in output folder) is what's actually run for each instance.  Currently I just blindly modify `PokemonRedExperiments/baselines/red_gym_env.py` when I need to and then copy-paste to the following because I cant be arsed to analyze how Eureka uses them yet:
	- Eureka\PokemonRedExperiments\baselines\red_gym_env.py (original file)
	- Eureka\eureka\envs\pokemon\red_gym_env_obs.py  (this appears to be what GPT "sees")
	- Eureka\eureka\envs\pokemon\red_gym_env.py
	- Eureka\PokemonRedExperiments\baselines\tasks\red_gym_envgpt.py
	- Eureka\isaacgymenvs\isaacgymenvs\tasks\red_gym_env.py


### Calling Structure Outline:

- call `python eureka.py env=red_gym_env sample=4 iteration=10 model=gpt-3.5-turbo-1106`
- `eureka/eureka.py` calls openAI to get a pack of code samples
- output request/response saved to e.g. `Eureka\eureka\outputs\eureka\2023-11-30_03-26-35\iter0_response0_to_4.txt`
- each code output is parsed to just the `compute_success()` function, appended to original code file (overwriting original function) and saved to e.g. `Eureka\eureka\outputs\eureka\2023-11-30_03-26-35\env_iter0_response0.py`
- each combined code sample is then run with process `PokemonRedExperiments/baselines/run_parallel_fast.py` which spawns pokemon game instance(s) (cpu_count=1 in run_parallel_fast.py) using the same code
- training session model is pulled from e.g. `Eureka\PokemonRedExperiments\sessions\session_28fb562c` (`session_4da05e87_main_good` is the original repo training) Currently just pulls latest session (which might be a different reward function.  TODO)
- output of each same is logged to e.g. `Eureka\eureka\outputs\eureka\2023-11-30_03-26-35\env_iter0_response0.txt` (this will combine logs from multi instances if cpu_count > 1. debatable if we want that).  This is where you're most likely to catch error logs when things break.  Expect a good half of all sample code to crash here even when things work (usually from GPT not fully understanding what it's allowed to write).  This improves with evolution.
- (the rest of output folder doesn't really do anything currently, more just for compatibility checks)
- a session is created in e.g. `Eureka\PokemonRedExperiments\sessions\session_28fb562c` with a zip file containing the meat, updated as code progresses.  This can be continued from in later runs (and currently the latest one is automatically)
- after instance completes its steps (ep_length = 256 * 100 in run_parallel_fast.py, keep it a multiple of 8(?)) the .txt output is parsed by `eureka.py`
- the whole output file is reduced to 10-ish snapshots over time and min/max/mean
- the total task success is measured by `tensorboard_logs['consecutive_successes'] = [max(tensorboard_logs['sum']) // min(tensorboard_logs['sum'])]`... a quite terrible function actually...  (Edit: updated to `tensorboard_logs['consecutive_successes'] = [tensorboard_logs['sum'][-1] // (max(tensorboard_logs['sum']) - min(tensorboard_logs['sum']))]` to be mildly better).  We want this to take the final score and normalize it according to the scale of rewards so the winner isn't just whichever code assigns bigger rewards.  Ideally should be the velocity of reward growth, normalized across all possible ways to cheat.  But to be fair, original Eureka seems to allow this kind of cheating and ChatGPT compensates?  Better might be to write this whole thing to be more in the style of original Eureka and run many many pokemon instances, tracking success measures for each one, and just use the total success of *those* to measure success of a code mutation.  Oh well. TODO.
- Anyway, eureka.py takes these for each sample code mutation, compares them and picks the "best", and feeds that into next GPT prompt along with the statistics it output.  Next code iterations therefore build off previous output and code.  Loop goes on!
- When eureka hits max `iteration=10` of this it completes, tries to analyze and run the best loop, and likely crashes lol (havent bothered to fix that part).  just a victory lap anyway.
- if you now want to re-run this all from latest progress, you'll want to manually replace `red_gym_env.py` everywhere with the final best version (just copy e.g. `env_iter10_response0.py` contents) and make sure its session is the latest in the `PokemonRedExperiments/sessions` folder
- whew, that's a lot to track and explain.  probably missing something. this is why nobody documents anything.  coulda just rewritten the whole architecture to make intuitive sense by now


 
# Eureka: Human-Level Reward Design via Coding Large Language Models

<div align="center">

[[Website]](https://eureka-research.github.io)
[[arXiv]](https://arxiv.org/abs/2310.12931)
[[PDF]](https://eureka-research.github.io/assets/eureka_paper.pdf)

[![Python Version](https://img.shields.io/badge/Python-3.8-blue.svg)](https://github.com/eureka-research/Eureka)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/eureka-research/Eureka)](https://github.com/eureka-research/Eureka/blob/main/LICENSE)
______________________________________________________________________

https://github.com/eureka-research/Eureka/assets/21993118/1abb960d-321a-4de9-b311-113b5fc53d4a


# PokemonRedExperiments: ML + PokemonRed
  
<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="https://github.com/PWhiddy/PokemonRedExperiments/blob/master/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="https://github.com/PWhiddy/PokemonRedExperiments/blob/master/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
