from mirideep.core import MiriDeepSpec

MDS = MiriDeepSpec(source='mylup',save_intermediate=True,standard='jena2',rrs={'ch1':1.4,'ch2':1.3,'ch3':1.2,'ch4':1.1},
	               bg_types={'ch1':'nod','ch2':'nod','ch3':'nod','ch4':'nod'},wave_correct=True,ch1_standard='hd163466_0723')
MDS.run_extract()
