import os
import subprocess
from mirideep.reduce_script import reduce
import logging

proposal_id = '1584'
observations = [{'dir':'data_as205', 'target_short':'as205n', 'target_name':'AS-205-N'},
                {'dir':'data_as209', 'target_short':'as209', 'target_name':'AS-209'},
                {'dir':'data_doar25', 'target_short':'doar25', 'target_name':'DOAR-25'},
                {'dir':'data_doar33', 'target_short':'doar33', 'target_name':'DOAR-33'},
                {'dir':'data_elias20', 'target_short':'elias20', 'target_name':'ELIAS-2-20'},
                {'dir':'data_elias24', 'target_short':'elias24', 'target_name':'ELIA-2-24'},
                {'dir':'data_elias27', 'target_short':'elias27', 'target_name':'ELIA-2-27'},
                {'dir':'data_hd142666', 'target_short':'hd142666', 'target_name':'HD-142666'},
                {'dir':'data_hd143006', 'target_short':'hd143006', 'target_name':'HD-143006'},
                {'dir':'data_hd163296', 'target_short':'hd163296', 'target_name':'HD-163296'},
                {'dir':'data_htlup', 'target_short':'htlup', 'target_name':'HT-LUP'},
                {'dir':'data_mylup', 'target_short':'mylup', 'target_name':'MY-LUP'},
                {'dir':'data_rulup', 'target_short':'rulup', 'target_name':'RU-LUP'},
                {'dir':'data_sr4', 'target_short':'sr4', 'target_name':'SR-4'},
                {'dir':'data_sz114', 'target_short':'sz114', 'target_name':'SZ-114'},
                {'dir':'data_sz129', 'target_short':'sz129', 'target_name':'SZ-129'},
                {'dir':'data_wsb52', 'target_short':'wsb52', 'target_name':'WSB-52'}]


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
