## Access argon
`ssh -o TCPKeepAlive=no -o ServerAliveInterval=30 -p 40 hawkid@argon.hpc.uiowa.edu`

- `-o TCPKeepAlive=no -o ServerAliveInterval=30` keeps the client from disconnecting if the script is taking a little while in interactive mode
- `-p 40 hawkid@` is only strictly necessary off-campus, but never hurts

## Use the UI queue
`qlogin -q UI`

- Without this it's impossible to run finetune.py in interactive mode

## Download libraries for finetuning
`module load py-pip`\
`bash pip_boilerplate.sh`

- `module load py-pip` allows you to use pip in the node
- `pip_boilerplate.sh` is included in this folder, it installs all the libraries necessary to run the finetuning script

## Submit a job to the UI queue
`qsub -q UI nameofyourjob.job`

- `-q UI` isn't necessary but I think it runs faster
- A .job file is just a bash script telling Argon what script to run.

## Check on your jobs
`qstat -u hawkid`

## Delete a job
`qdel -j job-ID`

- `job-ID` is the numerical ID for each job, assigned upon submitting a job, and can be also found with `qstat`

## Run a python script in interactive mode
`python3 finetune.py`

- `python3` is necessary as just `python` uses Python 2. The version of Python 3 used by default is also somewhat old, I think like 3.6, so a few newer features like ```:=``` don't work


