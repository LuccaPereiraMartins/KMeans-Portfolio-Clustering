
select * from raw.market_data 
where ticker = 'AGR'
limit 20

-- pre processing calculations previously done in python

SELECT
    a.*
    , as rsi
from raw.market_data as a