init:
	conda env create --name "gtable" -f gtable.yml
test:
	nosetests tests
