# Loss Lens

LossLens is a comprehensive visual analytics framework provides an interactive, multi-scale visual representation and analytical pipeline for exploring loss landscapes of
machine learning models.

## Run via Docker

To run the system through Docker, run the following under the root directory:

```
docker-compose up
```

## Run Locally

Running locally needs a bit more step. Before installing the LossLens, make sure you have [Nodejs](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-22-04), [Python](https://www.python.org), and [MongoDB](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/) installed first. Version requirements:

| Software | Min. Version |
| -------- | ------------ |
| Nodejs   | 18.0         |
| Python   | 3.8          |
| MongoDB  | 6.0          |

**Install GUI Dependencies**

Under the `gui` folder, run

```bash
npm install
```

to install all dependencies.

**Install Back-end Dependencies**

Under the `api` folder, run

```bash
pip install -r requirements.txt
pip install git+https://github.com/fra31/auto-attack
pip install git+https://github.com/RobustBench/robustbench.git
```

to install all dependencies.

**Restore the Data on Local Machine**

Under the `database` folder, run

```bash
mongorestore --host mongodb --port 27017 --db losslensdb ./losslensdb/
```

**Run LossLens**
Run

```bash
npm run dev
```

to start the frontend server.

Run

```bash
python server.py
```

to start the backend server.

## Adding case study

please see

## Contributing

We are welcoming contributions from our collaborators. We strongly encourage contributors to work in their own forks of this project. You can edit the code on your own branch after you create your own forks. When you are ready to contribute, you can click the `Compare & pull` request button next to your own branch under your repository.

This project highly recommand committers to sign their code using the [Developer Certificate of Origin (DCO)](https://developercertificate.org/) for each git commit but this is not required.
