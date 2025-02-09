dry_run:
	uv run python main.py 'saving=null' 'training.max_steps=10'

train:
	uv run python main.py
