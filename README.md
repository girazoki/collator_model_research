# collator_model_research

virtualenv -p python3 .

source bin/activate

pip install -r requirements.txt

python3 main_stall.py --active_collators 100 --stake_steps 0.01 --stall_steps 0.1 --difficulty_parameter 0.7 --number_of_trials 1000 --algorithm babe+aura
