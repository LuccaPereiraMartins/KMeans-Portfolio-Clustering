-- Active: 1738920736186@@127.0.0.1@5432@postgres@raw

drop table if exists raw.market_data;
create table raw.market_data(
    "date" timestamp not null,
    ticker varchar(8) not null,
    adj_close float not null,
    close float not null,
    high float not null,
    low float not null,
    open float not null,
    volume int not null,
    primary key ("date",ticker)
);
