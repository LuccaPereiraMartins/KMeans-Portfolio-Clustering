
select * from raw.market_data 
where ticker = 'AGR'
limit 20

-- pre processing calculations previously done in python

SELECT
    a.*
    , as rsi
from raw.market_data as a



-- given a ticker and a date range, extract the open and close price for that range

declare @startdate date
declare @enddate date
declare @ticker varchar(3)
select * 
from raw.market_data as md
where 
    md.date between @startdate and @enddate
    AND
    md.ticker = @ticker