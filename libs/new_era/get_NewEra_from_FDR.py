import sys
import requests

FDR='https://www.fdr.uni-hamburg.de/record/16738/files/'

def download_file(url, save_path):
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors

        # Open a local file with write-binary mode
        with open(save_path, 'wb') as file:
            # Write the content to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print('download successful:', save_path)
    except requests.exceptions.RequestException as e:
        print('Error for:', e)


def make_NewEra_filename(teff=None, logg=None, zscale=None, alpha_scale=None):
        """
         generate the (LTE) NewEra HSR filename to pass to the download code
        """
#
# now we need to fix the job name, file names etc in the target.
# 0. construct file names etc. from the target_nml.
# 1. read the new file.
# 2. use the re engine to change the data in RAM
# 3. write the data back to the file.
#


        if(zscale != 0.0):
          if(alpha_scale == 0.0):
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+f'{zscale:0=+4.1f}'
          else:
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+f'{zscale:0=+4.1f}'+'.alpha='+f'{alpha_scale:0=+3.1f}'
        else:
          if(alpha_scale == 0.0):
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+'-'+f'{zscale:0=3.1f}'
          else:
            job_name = 'lte'+f'{teff:0=5.0f}'+f'{-logg:3.2f}'+'-'+f'{zscale:0=3.1f}'+'.alpha='+f'{alpha_scale:0=+3.1f}'

        new_name = job_name+'.PHOENIX-NewEra-ACES-COND-2023.HSR.h5'

        #new_name = re.sub('lte','nlte', new_name.rstrip())
        #job_name = re.sub('lte','nlte', job_name.rstrip())

        return new_name

# testing:
url = 'https://www.fdr.uni-hamburg.de/record/16728/files/lte02300-0.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5?download=1'
save_path = './libs/new_era/lte02300-0.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5'
download_file(url, save_path)

n_args = len(sys.argv)

if(n_args <= 3): 
   print('usage: get_NewEra_from_FDR.py teff logg zscale [alpha_scale]')
   sys.exit()

Teff = float(sys.argv[1])
logg = float(sys.argv[2])
zscale = float(sys.argv[3])
if(n_args > 4): alpha_scale = float(sys.argv[4])
else: alpha_scale = 0.0

target = make_NewEra_filename(teff=Teff, logg=logg, zscale=zscale, alpha_scale=alpha_scale)

url = FDR+target+'?download=1'
save_file = './'+target

#print(url)
#print(save_file)
download_file(url, save_file)
