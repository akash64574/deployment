.PHONY: start install develop test

start:
	@bash bin/run.sh

install:
	@pip install -r requirements/common.txt

develop:
	@pip install -r requirements/common.txt -r requirements/develop.txt

test:
	@pytest tests -p no:warnings --verbose

