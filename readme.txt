Current process:
1. Ingest data through yfinance api
2. Store as local csv
3. Pre-process and reformat
4. Enrich with simple, daily financial metrics
5. Aggregate to monthly and enrich with Fama French factors
6. Cluster monthly groups with K-means++, initialising centroids to roughly
	follow momentum-based strategy
7. Visualise clusters with PCA
8. Compare and plot against FTSE250

Renewed process:
1. Ingest data through Alpha Vantage API
2. Store in Postgres database
3. Apply simple manipulations in SQL
	(or try to request via API)
4. Aggregate to monthly
	(or take monthly data)
5. Cluster with K-means or DBSCAN
{rest as above}