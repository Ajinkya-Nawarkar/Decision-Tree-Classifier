filename = 'data/weather.csv'

with open(filename, 'r') as f:
	with open('data/newweather.txt', 'w') as fw:
		data = f.readlines()

		for line in data:
			# nl = '[\'' + line[:-1] + '\']' + line[-1]
			# for i in nl:
			# 	if (i == ','):
			# 		nl = nl[:i] + '\'' + nl[i:]


			li = line.split(',') 
			
			fl = '['
			for each in li:
				fl += '\'' + each.strip('\n') + '\','

			fl = fl[:-1] + '],' + '\n'
			print fl

			fw.write(fl)

