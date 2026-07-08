import os
import subprocess
from mirideep.reduce_script import reduce
import logging

proposal_id = '1584'
observations = [{'dir':'data_mylup', 'target_short':'mylup', 'target_name':'MY-LUP'}]

# steps
run_dl    = True
run_step1 = False
run_step2 = True
run_step3 = True


# create logger 
logger = logging.getLogger('mirideep')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('mirideep.log', 'a')
fh.setFormatter(logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s'))

fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

for obs in observations:
	os.chdir(obs['dir'])

	try:
		reduce(target_short=obs['target_short'], target_name=obs['target_name'], proposal_id=proposal_id,
    	       run_dl=run_dl, run_step1=run_step1, run_step2=run_step2, run_step3=run_step3)
		logger.info('Successfully processed '+obs['target_short'])
	except:
		logger.info('Failed processing '+obs['target_short'])


	subprocess.run(["rm *_crf.fits"], shell=True)

	os.chdir('../')
