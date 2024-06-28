# The `GWAK` Pipeline
Here is the summary how to run `GWAK` algorithm training

## `Snakemake`

The code is organized using [`Snakemake`](https://snakemake.readthedocs.io/en/stable/).
The Snakemake workflow management system is a tool to create reproducible and scalable data analyses.

Check the [Snakefile](/.Snakefile) for the definition of analysis steps, eg `rules`.
Typically, each `rule` is a wrapper around a python function. Though, `Snakemake` can be used in much more cases.


To run snakemake do
```
snakemake -c1 {rule_name}
```
where `-c1` specifies number of cores provided (one in this case).
It became required to specify it in the latest versions of `Snakemake`,
so to make life easier you can add
`alias snakemake="snakemake -c1"` to your `bash/zsch/whatever` profile
and afterwards simply run `snakemake {rule_name}`.

If you want to run a rule, but Snakemake tells you `Nothing to be done`, use `-f`
to force it. Use `-F` to also force all the upstream rules to be re-run.

## Running `Snakemake` with Condor

In order to be able to submit jobs to `HTCondor`, install [snakemake-condor-profile](https://github.com/msto/snakemake-condor-profile).

Sending Snakemake process to `HTCondor`:

    $ snakemake --profile HTCondor