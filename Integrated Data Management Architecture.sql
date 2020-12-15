/* ######################## */
/* ######### CRM ########## */
/* ######################## */


/* DROP SEQUENCES */
DROP SEQUENCE general_member_id_seq;
DROP SEQUENCE business_member_id_seq;
DROP SEQUENCE credit_card_id_seq;
DROP SEQUENCE costco_user_id_seq;

/* DROP TABLES */
DROP TABLE general_business_link;
DROP TABLE business_user_table;
DROP TABLE general_user_table;

/* CREATE SEQUENCES */
CREATE SEQUENCE general_member_id_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

CREATE SEQUENCE business_member_id_seq
START WITH 1000001 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

CREATE SEQUENCE costco_user_id_seq
START WITH 1000001 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

CREATE SEQUENCE credit_card_id_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

/* CREATE TABLES */
CREATE TABLE general_user_table (
    costco_user_id              NUMBER          DEFAULT costco_user_id_seq.NEXTVAL    PRIMARY KEY,
    general_member_id           NUMBER          DEFAULT general_member_id_seq.NEXTVAL,
    first_name                  VARCHAR(30)     NOT NULL,
    last_name                   VARCHAR(30)     NOT NULL,
    date_of_birth               DATE            NOT NULL,
    phone_number                CHAR(12)        NOT NULL,
    email                       VARCHAR(50)     NOT NULL    UNIQUE,
    acct_username               VARCHAR(30)     NOT NULL,
    acct_password               VARCHAR(30)     NOT NULL,
    mailing_address             VARCHAR(100)    NOT NULL,
    city                        VARCHAR(30)     NOT NULL,
    state_name                  CHAR(2)         NOT NULL,
    country                     VARCHAR(30)     NOT NULL,
    zip                         CHAR(5)         NOT NULL,
    credit_card_id              NUMBER          DEFAULT credit_card_id_seq.NEXTVAL,
    card_number                 NUMBER          NOT NULL,
    card_type                   VARCHAR(50)     NOT NULL,
    expiration_date             DATE            NOT NULL,
    security_code               NUMBER          NOT NULL,
    billing_city                VARCHAR(50)     NOT NULL,
    billing_state               CHAR(2)         NOT NULL,
    billing_zip                 CHAR(5)         NOT NULL
);

CREATE TABLE business_user_table (
    costco_user_id              NUMBER          DEFAULT costco_user_id_seq.NEXTVAL  PRIMARY KEY,
    business_member_id          NUMBER          DEFAULT business_member_id_seq.NEXTVAL,
    first_name                  VARCHAR(30)     NOT NULL,
    last_name                   VARCHAR(30)     NOT NULL,
    date_of_birth               DATE            NOT NULL,
    phone_number                CHAR(12)        NOT NULL,
    email                       VARCHAR(50)     NOT NULL    UNIQUE,
    acct_username               VARCHAR(30)     NOT NULL,
    acct_password               VARCHAR(30)     NOT NULL,
    mailing_address             VARCHAR(100)    NOT NULL,
    city                        VARCHAR(30)     NOT NULL,
    state_name                  CHAR(2)         NOT NULL,
    country                     VARCHAR(30)     NOT NULL,
    zip                         CHAR(5)         NOT NULL,
    credit_card_id              NUMBER          DEFAULT credit_card_id_seq.NEXTVAL,
    card_number                 NUMBER          NOT NULL,
    card_type                   VARCHAR(50)     NOT NULL,
    expiration_date             DATE            NOT NULL,
    security_code               NUMBER          NOT NULL,
    billing_city                VARCHAR(50)     NOT NULL,
    billing_state               CHAR(2)         NOT NULL,
    billing_zip                 CHAR(5)         NOT NULL,
    business_name               VARCHAR(50)     NOT NULL,
    business_street             VARCHAR(100)    NOT NULL,
    business_zip                CHAR(5)         NOT NULL,
    business_city               VARCHAR(30)     NOT NULL,
    business_state              CHAR(2)         NOT NULL,
    business_email              VARCHAR(50)     NOT NULL    UNIQUE,
    business_phone              CHAR(12)        NOT NULL
);

CREATE TABLE general_business_link (

    general_costco_id           NUMBER          NOT NULL,
    business_costco_id          NUMBER          NOT NULL,
    
    CONSTRAINT general_business_cpk             PRIMARY KEY (general_costco_id, business_costco_id),
    CONSTRAINT general_costco_id_fk             FOREIGN KEY (general_costco_id) REFERENCES general_user_table (costco_user_id),
    CONSTRAINT business_costco_id               FOREIGN KEY (business_costco_id) REFERENCES business_user_table (costco_user_id)
);

/* Seed data */
--Insert users into general_user_table:
INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Joe', 'Niehaus', TO_DATE('01/01/1990', 'DD/MM/YYYY'), '5342345345', 'joe@email.com', 'joe', 'lawnmower', '321 hickory', 'austin', 'TX', 'USA', '78741', 1870779127164844, 'AMEX', TO_DATE('01/01/2024','DD/MM/YYYY'), 234, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('John', 'Newman', TO_DATE('02/02/1991', 'DD/MM/YYYY'), '3546345663', 'john@email.com', 'john', 'nothing', '234 west', 'austin', 'TX', 'USA', '78741', 1299651297163822, 'MASTERCARD', TO_DATE('02/02/2024','DD/MM/YYYY'), 235, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Jack', 'Sparrow', TO_DATE('03/03/1992', 'DD/MM/YYYY'), '2345234524', 'jack@email.com', 'jack', 'rum', '345 east', 'austin', 'TX', 'USA', '78741', 2099033307384861, 'VISA', TO_DATE('03/03/2024','DD/MM/YYYY'), 543, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Christian', 'Lee', TO_DATE('04/04/1993', 'DD/MM/YYYY'), '5787654456', 'christian@email.com', 'christian', 'beer', '124 north', 'austin', 'TX', 'USA', '78741', 1229817149036205, 'AMEX', TO_DATE('04/04/2024','DD/MM/YYYY'), 243, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Christ', 'Almighty', TO_DATE('05/05/1994', 'DD/MM/YYYY'), '3868386635', 'christ@email.com', 'jesus', 'wine', '763 dean keeton', 'austin', 'TX', 'USA', '78741', 1421862339481545, 'AMEX', TO_DATE('05/05/2024','DD/MM/YYYY'), 687, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Matthew', 'Streichler', TO_DATE('06/06/1995', 'DD/MM/YYYY'), '8067844567', 'strike@email.com', 'strike', 'soccer', '654 mlk', 'austin', 'TX', 'USA', '78741', 1421916444582234, 'VISA', TO_DATE('06/06/2024','DD/MM/YYYY'), 897, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Mona', 'Lisa', TO_DATE('07/07/1996', 'DD/MM/YYYY'), '2456325467', 'mona@email.com', 'mona', 'painting', '657 manor', 'austin', 'TX', 'USA', '78741', 1678343277176177, 'MASTERCARD', TO_DATE('07/07/2024','DD/MM/YYYY'), 465, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Mike', 'Jackson', TO_DATE('08/08/1997', 'DD/MM/YYYY'), '6383586356', 'mike@email.com', 'mike', 'singing', '685 first', 'austin', 'TX', 'USA', '78741', 1948035792112575, 'AMEX', TO_DATE('08/08/2024','DD/MM/YYYY'), 345, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Pete', 'Davis', TO_DATE('09/09/1998', 'DD/MM/YYYY'), '1235617436', 'pete@email.com', 'pete', 'golf', '976 lamar', 'austin', 'TX', 'USA', '78741', 1499728192356417, 'VISA', TO_DATE('09/09/2024','DD/MM/YYYY'), 867, 'Austin', 'TX', 78741);

INSERT INTO general_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip)
VALUES ('Pistol', 'Maravich', TO_DATE('10/10/1999', 'DD/MM/YYYY'), '3573568652', 'pistol@email.com', 'pistol', 'bball', '455 congress', 'austin', 'TX', 'USA', '78741', 1175929133238428, 'VISA', TO_DATE('01/11/2024','DD/MM/YYYY'), 456, 'Austin', 'TX', 78741);

--Insert businesses into business_user_table:
INSERT INTO business_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone)
VALUES ('Joe', 'Niehaus', TO_DATE('01/01/1990', 'DD/MM/YYYY'), '5342345345', 'joe@email.com', 'joe', 'lawnmower', '321 hickory', 'austin', 'TX', 'USA', '78741', 1870779127164844, 'AMEX', TO_DATE('01/01/2024','DD/MM/YYYY'), 234, 'Austin', 'TX', 78741, 'Jacks Rum Co', '5432 Jones', 78741, 'Austin', 'TX','jrumco@biz.com', 7980987654);

INSERT INTO business_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone)
VALUES ('Christian', 'Lee', TO_DATE('04/04/1993', 'DD/MM/YYYY'), '5787654456', 'christian@email.com', 'christian', 'beer', '124 north', 'austin', 'TX', 'USA', '78741', 1229817149036205, 'AMEX', TO_DATE('04/04/2024','DD/MM/YYYY'), 243, 'Austin', 'TX', 78741, 'Water2Wine', '5123 Jesus', 78741, 'Austin', 'TX','water2wine@biz.com', 1243553547);

INSERT INTO business_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone)
VALUES ('Matthew', 'Streichler', TO_DATE('06/06/1995', 'DD/MM/YYYY'), '8067844567', 'strike@email.com', 'strike', 'soccer', '654 mlk', 'austin', 'TX', 'USA', '78741', 1421916444582234, 'VISA', TO_DATE('06/06/2024','DD/MM/YYYY'), 897, 'Austin', 'TX', 78741, 'Mikes Records', '112 Congress', 78741, 'Austin', 'TX','recordshop@biz.com', 1235235543);

INSERT INTO business_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone)
VALUES ('Pete', 'Davis', TO_DATE('09/09/1998', 'DD/MM/YYYY'), '1235617436', 'pete@email.com', 'pete', 'golf', '976 lamar', 'austin', 'TX', 'USA', '78741', 1499728192356417, 'VISA', TO_DATE('09/09/2024','DD/MM/YYYY'), 867, 'Austin', 'TX', 78741, 'Austin Golf Club', '423 Lakeway', 78741, 'Austin', 'TX','agc@biz.com', 8567883456);

INSERT INTO business_user_table(first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone)
VALUES ('Mike', 'Jackson', TO_DATE('08/08/1997', 'DD/MM/YYYY'), '6383586356', 'mike@email.com', 'mike', 'singing', '685 first', 'austin', 'TX', 'USA', '78741', 1948035792112575, 'AMEX', TO_DATE('08/08/2024','DD/MM/YYYY'), 345, 'Austin', 'TX', 78741, 'Pistols Gym', '3245 Riverside', 78741, 'Austin', 'TX','gymrats@biz.com', 1234552645);


/* ######################## */
/* ######### SCM ########## */
/* ######################## */

/* DROP SEQUENCES */
DROP SEQUENCE vendor_id_seq;
DROP SEQUENCE invoice_id_seq;
DROP SEQUENCE inventory_id_seq;
DROP SEQUENCE costco_scm_id_seq;

/* DROP TABLES */
DROP TABLE vendor_invoice_inventory_link;
DROP TABLE vendor_invoice_table;
DROP TABLE costco_inventory_table;

/* CREATE SEQUENCES */
CREATE SEQUENCE vendor_id_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

CREATE SEQUENCE invoice_id_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

CREATE SEQUENCE inventory_id_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

CREATE SEQUENCE costco_scm_id_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999;

/* CREATE TABLES */
CREATE TABLE vendor_invoice_table (

    costco_scm_id               NUMBER          DEFAULT costco_scm_id_seq.NEXTVAL   PRIMARY KEY,
    vendor_id                   NUMBER          DEFAULT vendor_id_seq.NEXTVAL,
    vendor_name                 VARCHAR(30)     NOT NULL,
    street                      VARCHAR(100)    NOT NULL,
    city                        VARCHAR(30)     NOT NULL,
    state_name                  CHAR(2)         NOT NULL,
    country                     VARCHAR(30)     NOT NULL,
    zip                         CHAR(5)         NOT NULL,
    phone_number                CHAR(12)        NOT NULL,
    invoice_id                  NUMBER          DEFAULT invoice_id_seq.NEXTVAL,
    invoice_number              NUMBER          NOT NULL,
    invoice_date                DATE            NOT NULL,
    invoice_total               NUMBER          NOT NULL,
    invoice_due_date            DATE            NOT NULL,
    payment_date                DATE            NOT NULL
);

CREATE TABLE costco_inventory_table (

    costco_scm_id               NUMBER          DEFAULT costco_scm_id_seq.NEXTVAL   PRIMARY KEY,
    inventory_id                NUMBER          DEFAULT inventory_id_seq.NEXTVAL,
    vendor_id                   NUMBER          DEFAULT vendor_id_seq.NEXTVAL,
    item                        VARCHAR(50)     NOT NULL,
    cost_p_unit                 NUMBER          NOT NULL,
    quantity                    NUMBER          NOT NULL,
    low_threshold               NUMBER          NOT NULL
);

CREATE TABLE vendor_invoice_inventory_link (

    vendor_invoice_id           NUMBER          NOT NULL,
    costco_inventory_id         NUMBER          NOT NULL,
    
    CONSTRAINT vendor_invoice_inventory_cpk     PRIMARY KEY (vendor_invoice_id, costco_inventory_id),
    CONSTRAINT vendor_invoice_id_fk             FOREIGN KEY (vendor_invoice_id) REFERENCES vendor_invoice_table (costco_scm_id),
    CONSTRAINT costco_inventory_id_fk           FOREIGN KEY (costco_inventory_id) REFERENCES costco_inventory_table (costco_scm_id)
);

/* Seed the data */
--Insert vendors into vendor_table;
INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('jasons seafood', '45th', 'Austin', 'TX', 'USA', 78741, 1486403433, 4210, 20580, sysdate-30, sysdate+30, sysdate+30);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('jordans bikeshop', '51st', 'Austin', 'TX', 'USA', 78741, 5798196626, 817891, 13198.5, sysdate-20, sysdate+30, sysdate);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('joshs surfshop', '32nd', 'Austin', 'TX', 'USA', 78741, 1178391986, 377, 2527.8, sysdate-17, sysdate+30, sysdate+30);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('joes gunshop', 'lamar', 'Austin', 'TX', 'USA', 78741, 3815606998, 982, 36015, sysdate-6, sysdate+30, sysdate+30);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('matts games', 'congress', 'Austin', 'TX', 'USA', 78741, 6886036772, 100772, 1035.15, sysdate-12, sysdate+30, sysdate);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('peters bagels', 'ben white', 'Austin', 'TX', 'USA', 78741, 2983413591, 88709, 3210, sysdate-22, sysdate+30, sysdate+30);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('Tejs Books', '38th', 'Austin', 'TX', 'USA', 78741, 6770909118, 56, 20401.98, sysdate-40, sysdate+30, sysdate);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('Bens Beauty Supplies', 'tom green', 'Austin', 'TX', 'USA', 78741, 8850856384, 10044, 312, sysdate-2, sysdate+30, sysdate+30);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('Dians Home Supplies', '50th', 'Austin', 'TX', 'USA', 78741, 8854465352, 118753, 6300.03, sysdate-3, sysdate+30, sysdate);

INSERT INTO vendor_invoice_table(vendor_name, street, city, state_name, country, zip, phone_number, invoice_number, invoice_total, invoice_date, invoice_due_date, payment_date)
VALUES ('Davids Paper Products', 'frontage', 'Austin', 'TX', 'USA', 78741, 2746135000, 4605, 11009.90, sysdate-7, sysdate+30, sysdate+30);

--Insert inventory into costco_inventory_table;
INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000000, 'Bulgarian Sturgeon Caviar', 51.45, 400, 15);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000001, 'Beach Cruiser Bicycle', 175.98, 75, 10);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000002, '9ft Foam Long Board', 210.65, 12, 2);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000003, 'Case Target Ammo', 120.05, 300, 100);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000004, 'Hand Painted Collegiate Cornhole Set', 69.01, 15, 1);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000005, 'Stay-Fresh Everything Bagels (half-dozen)', 3.21, 1000, 0);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000006, 'Vintage Encyclopedia Set', 10200.99, 2, 0);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000007, '12-pack Masks', 2.08, 450, 100);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000008, '6-Person Hot Tub', 2100.01, 3, 0);

INSERT INTO costco_inventory_table(vendor_id, item, cost_p_unit, quantity, low_threshold)
VALUES (1000009, 'Photo Printer Paper (pallet)', 1100.99, 10, 5);


/* ######################## */
/* ######## Sales ######### */
/* ######################## */

/* DROP SEQUENCES */
DROP SEQUENCE instore_id_seq;
DROP SEQUENCE online_id_seq;

/* DROP TABLES */
DROP TABLE instore_online_link;
DROP TABLE online_order_table;
DROP TABLE instore_order_table;

/* CREATE SEQUENCES */
CREATE SEQUENCE instore_id_seq
START WITH 1000000 INCREMENT BY 2
MINVALUE 1000000 MAXVALUE 9999999;

CREATE SEQUENCE online_id_seq
START WITH 1000001 INCREMENT BY 2
MINVALUE 1000000 MAXVALUE 9999999;

/* CREATE TABLES */
CREATE TABLE instore_order_table (
    order_id                    NUMBER          DEFAULT instore_id_seq.NEXTVAL    PRIMARY KEY,
    inventory_id                NUMBER          NOT NULL,
    item_name                   VARCHAR(100)    NOT NULL,
    item_price                  NUMBER          NOT NULL,
    item_quantity               NUMBER          NOT NULL,
    date_of_purch               DATE            DEFAULT SYSDATE,
    credit_card_id              NUMBER(7)       NOT NULL
);

CREATE TABLE online_order_table (
    order_id                    NUMBER          DEFAULT online_id_seq.NEXTVAL    PRIMARY KEY,
    inventory_id                NUMBER          NOT NULL,
    item_name                   VARCHAR(100)    NOT NULL,
    item_price                  NUMBER          NOT NULL,
    item_quantity               NUMBER          NOT NULL,
    date_of_purch               DATE            DEFAULT SYSDATE,
    credit_card_id              NUMBER(7)       NOT NULL 
);

CREATE TABLE instore_online_link (

    instore_id                  NUMBER          NOT NULL,
    online_id                   NUMBER          NOT NULL,
    
    CONSTRAINT in_on_cpk                        PRIMARY KEY (instore_id, online_id),
    CONSTRAINT in_id_fk                         FOREIGN KEY (instore_id) REFERENCES instore_order_table (order_id),
    CONSTRAINT on_id_fk                         FOREIGN KEY (online_id) REFERENCES online_order_table (order_id)
);

/* Seed data */
--INSERT orders into instore_order;
INSERT INTO instore_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000004,'Hotdogs (pack of 8)',4.99,3,TO_DATE('08/08/2020', 'DD/MM/YYYY'),1000002);

INSERT INTO instore_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000002,'Hotdog Buns (pack of 6)',2.95,4,TO_DATE('14/08/2020', 'DD/MM/YYYY'),1000001);

INSERT INTO instore_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000003,'Free Range Whole Organic Turkey (17lb)',119.99,1,TO_DATE('16/08/2020', 'DD/MM/YYYY'),1000009);

INSERT INTO instore_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000004,'Hotdogs (pack of 8)',4.99,15,TO_DATE('27/08/2020', 'DD/MM/YYYY'),1000006);

INSERT INTO instore_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000000,'Whole Bean Coffee (5 lb)',41.99,1,TO_DATE('04/09/2020', 'DD/MM/YYYY'),1000005);

--INSERT orders into online_order;
INSERT INTO online_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000001,'6-Person Hot Tub',2999.99,1,TO_DATE('06/08/2020', 'DD/MM/YYYY'),1000007);

INSERT INTO online_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000000,'Dual Stage Snow Blower',1199.99,4,TO_DATE('10/08/2020', 'DD/MM/YYYY'),1000007);

INSERT INTO online_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000002,'26" Beach Cruiser Bicycle',249.99,2,TO_DATE('14/08/2020', 'DD/MM/YYYY'),1000006);

INSERT INTO online_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000003,'Split King Memory Foam Mattress w/ Base',3499.99,1,TO_DATE('28/08/2020', 'DD/MM/YYYY'),1000007);

INSERT INTO online_order_table(inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id)
VALUES (1000000,'Bulgarian Sturgeon Caviar (2 oz)',69.95,3,TO_DATE('01/09/2020', 'DD/MM/YYYY'),1000004);


/* ######################## */
/* #### Data Warehouse #### */
/* ######################## */

/* DROP TABLES */
DROP TABLE order_table_dw;
DROP TABLE costco_inventory_table_dw;
DROP TABLE invoice_table_dw;
DROP TABLE vendor_table_dw;
DROP TABLE credit_card_dw;
DROP TABLE business_membership_dw;
DROP TABLE general_membership_dw;
DROP TABLE costco_user_table_dw;

/* CREATE TABLES */
CREATE TABLE costco_user_table_dw AS (
    SELECT costco_user_id, first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, 'General' AS data_source
    FROM general_user_table
    UNION
    SELECT costco_user_id, first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, 'Business' AS data_source
    FROM business_user_table
);
ALTER TABLE costco_user_table_dw
ADD CONSTRAINT costco_user_id_pk_dw PRIMARY KEY (costco_user_id);

CREATE TABLE general_membership_dw AS (
    SELECT general_member_id, costco_user_id
    FROM general_user_table
);
ALTER TABLE general_membership_dw
ADD CONSTRAINT general_member_id_pk_dw1 PRIMARY KEY (general_member_id)
ADD CONSTRAINT costco_user_id_fk1_dw FOREIGN KEY (costco_user_id) REFERENCES costco_user_table_dw (costco_user_id);

CREATE TABLE business_membership_dw AS (
    SELECT business_member_id, costco_user_id, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone
    FROM business_user_table
);
ALTER TABLE business_membership_dw
ADD CONSTRAINT business_member_id_pk_dw PRIMARY KEY (business_member_id)
ADD CONSTRAINT costco_user_id_fk_dw FOREIGN KEY (costco_user_id) REFERENCES costco_user_table_dw (costco_user_id);

CREATE TABLE credit_card_dw AS (
    SELECT credit_card_id, costco_user_id, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, 'General' AS data_source
    FROM general_user_table
    UNION
    SELECT credit_card_id, costco_user_id, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, 'Business' AS data_source
    FROM business_user_table
);
ALTER TABLE credit_card_dw
ADD CONSTRAINT credit_card_id_pk_dw PRIMARY KEY (credit_card_id)
ADD CONSTRAINT costco_user_id_fk_dw1 FOREIGN KEY (costco_user_id) REFERENCES costco_user_table_dw (costco_user_id);

CREATE TABLE vendor_table_dw AS (
    SELECT vendor_id, vendor_name, street, city, state_name, country, zip, phone_number
    FROM vendor_invoice_table
);
ALTER TABLE vendor_table_dw
ADD CONSTRAINT vendor_id_pk_dw PRIMARY KEY (vendor_id);

CREATE TABLE invoice_table_dw AS (
    SELECT invoice_id, vendor_id, invoice_number, invoice_date, invoice_total, invoice_due_date, payment_date
    FROM vendor_invoice_table
);
ALTER TABLE invoice_table_dw
ADD CONSTRAINT invoice_id_pk_dw PRIMARY KEY (invoice_id)
ADD CONSTRAINT vendor_id_fk_dw1 FOREIGN KEY (vendor_id) REFERENCES vendor_table_dw (vendor_id);

CREATE TABLE costco_inventory_table_dw AS (
    SELECT inventory_id, vendor_id, item, cost_p_unit, quantity, low_threshold
    FROM costco_inventory_table
);
ALTER TABLE costco_inventory_table_dw
ADD CONSTRAINT inventory_id_pk_dw PRIMARY KEY (inventory_id)
ADD CONSTRAINT vendor_id_fk_dw2 FOREIGN KEY (vendor_id) REFERENCES vendor_table_dw (vendor_id);

CREATE TABLE order_table_dw AS (
    SELECT order_id, inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id, 'instore' AS data_source
    FROM instore_order_table
    UNION
    SELECT order_id, inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id, 'online' AS data_source
    FROM online_order_table
);
ALTER TABLE order_table_dw
ADD CONSTRAINT order_id_pk_dw PRIMARY KEY (order_id)
ADD CONSTRAINT cc_id_fk_dw FOREIGN KEY (credit_card_id) REFERENCES credit_card_dw (credit_card_id)
ADD CONSTRAINT inventory_id_fk_dw FOREIGN KEY (inventory_id) REFERENCES costco_inventory_table_dw (inventory_id);


/* ######################## */
/* ######### ETL ########## */
/* ######################## */

CREATE OR REPLACE PROCEDURE costco_etl_proc AS
BEGIN

    /* INSERT block #1 - costco_user_table_dw (from general_user_table) */
    INSERT INTO costco_user_table_dw (costco_user_id, first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, data_source)
    SELECT costco_user_id, first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, 'General' AS data_source
    FROM general_user_table gu
    WHERE NOT EXISTS (
        SELECT costco_user_id
        FROM costco_user_table_dw cu
        WHERE gu.costco_user_id = cu.costco_user_id
    );
    
    /* UPDATE block #1 - costco_user_table_dw (from general_user_table) */
    MERGE INTO costco_user_table_dw cu
        USING general_user_table gu
        ON (cu.costco_user_id = gu.costco_user_id)
        WHEN MATCHED THEN
            UPDATE SET
                cu.first_name = gu.first_name,
                cu.last_name = gu.last_name,
                cu.date_of_birth = gu.date_of_birth,
                cu.phone_number = gu.phone_number,
                cu.email = gu.email,
                cu.acct_username = gu.acct_username,
                cu.acct_password = gu.acct_password,
                cu.mailing_address = gu.mailing_address,
                cu.city = gu.city,
                cu.state_name = gu.state_name,
                cu.country = gu.country,
                cu.zip = gu.zip;
    
    /* INSERT block #2 - costco_user_table_dw (from business_user_table) */
    INSERT INTO costco_user_table_dw (costco_user_id, first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, data_source)
    SELECT costco_user_id, first_name, last_name, date_of_birth, phone_number, email, acct_username, acct_password, mailing_address, city, state_name, country, zip, 'Business' AS data_source
    FROM business_user_table bu
    WHERE NOT EXISTS (
        SELECT costco_user_id
        FROM costco_user_table_dw cu
        WHERE bu.costco_user_id = cu.costco_user_id
    );

    /* UPDATE block #2 - costco_user_table_dw (from business_user_table) */
    MERGE INTO costco_user_table_dw cu
        USING business_user_table bu
        ON (cu.costco_user_id = bu.costco_user_id)
        WHEN MATCHED THEN
            UPDATE SET
                cu.first_name = bu.first_name,
                cu.last_name = bu.last_name,
                cu.date_of_birth = bu.date_of_birth,
                cu.phone_number = bu.phone_number,
                cu.email = bu.email,
                cu.acct_username = bu.acct_username,
                cu.acct_password = bu.acct_password,
                cu.mailing_address = bu.mailing_address,
                cu.city = bu.city,
                cu.state_name = bu.state_name,
                cu.country = bu.country,
                cu.zip = bu.zip;

    /* INSERT block #3 - general_member_dw */
    INSERT INTO general_membership_dw (general_member_id, costco_user_id)
    SELECT general_member_id, costco_user_id
    FROM general_user_table gu
    WHERE NOT EXISTS (
        SELECT general_member_id
        FROM general_membership_dw gm
        WHERE gu.general_member_id = gm.general_member_id
    );
    
    /* UPDATE block #3 - general_member_dw */
    MERGE INTO general_membership_dw gm
        USING general_user_table gu
        ON (gu.general_member_id = gm.general_member_id)
        WHEN MATCHED THEN
            UPDATE SET
                gm.costco_user_id = gu.costco_user_id;

    /* INSERT block #4 - business_membership_dw */
    INSERT INTO business_membership_dw (business_member_id, costco_user_id, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone)
    SELECT business_member_id, costco_user_id, business_name, business_street, business_zip, business_city, business_state, business_email, business_phone
    FROM business_user_table bu
    WHERE NOT EXISTS (
        SELECT business_member_id
        FROM business_membership_dw bm
        WHERE bm.business_member_id = bu.business_member_id
    );

    /* UPDATE block #4 - business_membership_dw */
    MERGE INTO business_membership_dw bm
        USING business_user_table bu
        ON (bm.business_member_id = bu.business_member_id)
        WHEN MATCHED THEN
            UPDATE SET
                bm.costco_user_id = bu.costco_user_id, --> NOT SURE IF YOU CAN UPDATE A FK
                bm.business_name = bu.business_name,
                bm.business_street = bu.business_street,
                bm.business_zip = bu.business_zip,
                bm.business_city = bu.business_city,
                bm.business_state = bu.business_state,
                bm.business_email = bu.business_email,
                bm.business_phone = bu.business_phone;

    /* INSERT block #5 - credit_card_dw (from general_user_table) */
    INSERT INTO credit_card_dw (credit_card_id, costco_user_id, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, data_source)
    SELECT credit_card_id, costco_user_id, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, 'General' AS data_source
    FROM general_user_table gu
    WHERE NOT EXISTS (
        SELECT credit_card_id
        FROM credit_card_dw cc
        WHERE gu.credit_card_id = cc.credit_card_id
    );
    
    /* UPDATE block #5 - credit_card_dw (from general_user_table) */
    MERGE INTO credit_card_dw cc
        USING general_user_table gu
        ON (cc.credit_card_id = gu.credit_card_id)
        WHEN MATCHED THEN
            UPDATE SET
                cc.costco_user_id = gu.costco_user_id, --> NOT SURE IF YOU CAN UPDATE A FK
                cc.card_number = gu.card_number,
                cc.card_type = gu.card_type,
                cc.expiration_date = gu.expiration_date,
                cc.security_code = gu.security_code,
                cc.billing_city = gu.billing_city,
                cc.billing_state = gu.billing_state,
                cc.billing_zip = gu.billing_zip;
    
    /* INSERT block #6 - credit_card_dw (from business_user_table) */
    INSERT INTO credit_card_dw (credit_card_id, costco_user_id, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, data_source)
    SELECT credit_card_id, costco_user_id, card_number, card_type, expiration_date, security_code, billing_city, billing_state, billing_zip, 'Business' AS data_source
    FROM business_user_table bu
    WHERE NOT EXISTS (
        SELECT credit_card_id
        FROM credit_card_dw cc
        WHERE bu.credit_card_id = cc.credit_card_id
    );

    /* UPDATE block #6 - credit_card_dw (from business_user_table) */
    MERGE INTO credit_card_dw cc
        USING business_user_table bu
        ON (cc.credit_card_id = bu.credit_card_id)
        WHEN MATCHED THEN
            UPDATE SET
                cc.costco_user_id = bu.costco_user_id, --> NOT SURE IF YOU CAN UPDATE A FK
                cc.card_number = bu.card_number,
                cc.card_type = bu.card_type,
                cc.expiration_date = bu.expiration_date,
                cc.security_code = bu.security_code,
                cc.billing_city = bu.billing_city,
                cc.billing_state = bu.billing_state,
                cc.billing_zip = bu.billing_zip;

    /* INSERT block #7 - vendor_table_dw */
    INSERT INTO vendor_table_dw (vendor_id, vendor_name, street, city, state_name, country, zip, phone_number)
    SELECT vendor_id, vendor_name, street, city, state_name, country, zip, phone_number
    FROM vendor_invoice_table vit
    WHERE NOT EXISTS (
        SELECT vendor_id
        FROM vendor_table_dw vt
        WHERE vt.vendor_id = vit.vendor_id
    );
    
    /* UPDATE block #7 - vendor_table_dw */
    MERGE INTO vendor_table_dw vdw
        USING vendor_invoice_table vt
        ON (vdw.vendor_id = vt.vendor_id)
        WHEN MATCHED THEN
            UPDATE SET
                vdw.vendor_name = vt.vendor_name,
                vdw.street = vt.street,
                vdw.city = vt.city,
                vdw.state_name = vt.state_name,
                vdw.country = vt.country,
                vdw.zip = vt.zip,
                vdw.phone_number = vt.phone_number;
                
    /* INSERT block #8 - invoice_table_dw */
    INSERT INTO invoice_table_dw (invoice_id, vendor_id, invoice_number, invoice_date, invoice_total, invoice_due_date, payment_date)
    SELECT invoice_id, vendor_id, invoice_number, invoice_date, invoice_total, invoice_due_date, payment_date
    FROM vendor_invoice_table vit
    WHERE NOT EXISTS (
        SELECT invoice_id
        FROM invoice_table_dw it
        WHERE it.invoice_id = vit.invoice_id
    );
    
    /* UPDATE block #8 - invoice_table_dw */
    MERGE INTO invoice_table_dw idw
        USING vendor_invoice_table vt
        ON (idw.invoice_id = vt.invoice_id)
        WHEN MATCHED THEN
            UPDATE SET
                idw.vendor_id = vt.vendor_id, --> NOT SURE IF YOU CAN UPDATE A FK
                idw.invoice_number = vt.invoice_number,
                idw.invoice_date = vt.invoice_date,
                idw.invoice_total = vt.invoice_total,
                idw.invoice_due_date = vt.invoice_due_date,
                idw.payment_date = vt.payment_date;
    
    /* INSERT block #9 - costco_inventory_table_dw */
    INSERT INTO costco_inventory_table_dw (inventory_id, vendor_id, item, cost_p_unit, quantity, low_threshold)
    SELECT inventory_id, vendor_id, item, cost_p_unit, quantity, low_threshold
    FROM costco_inventory_table cit
    WHERE NOT EXISTS (
        SELECT inventory_id
        FROM costco_inventory_table_dw citd
        WHERE citd.inventory_id = cit.inventory_id
    );
   
    /* UPDATE block #9 - costco_inventory_table_dw */
    MERGE INTO costco_inventory_table_dw cidw
        USING costco_inventory_table cit
        ON (cidw.inventory_id = cit.inventory_id)
        WHEN MATCHED THEN
            UPDATE SET
                cidw.vendor_id = cit.vendor_id, --> NOT SURE IF YOU CAN UPDATE A FK
                cidw.item = cit.item,
                cidw.cost_p_unit = cit.cost_p_unit,
                cidw.quantity = cit.quantity,
                cidw.low_threshold = cit.low_threshold;
    
    /* INSERT block #10 - order_table_dw (from instore_order) */
    INSERT INTO order_table_dw (order_id, inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id, data_source)
    SELECT order_id, inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id, 'instore' AS data_source
    FROM instore_order_table io
    WHERE NOT EXISTS (
        SELECT order_id
        FROM order_table_dw ot
        WHERE ot.order_id = io.order_id
    );
    
    /* UPDATE block #10 - order_table_dw (from instore_order) */
    MERGE INTO order_table_dw odw
        USING instore_order_table i
        ON (odw.order_id = i.order_id)
        WHEN MATCHED THEN
            UPDATE SET
                odw.inventory_id = i.inventory_id, --> NOT SURE IF YOU CAN UPDATE A FK
                odw.item_name = i.item_name,
                odw.item_price = i.item_price,
                odw.item_quantity = i.item_quantity,
                odw.date_of_purch = i.date_of_purch,
                odw.credit_card_id = i.credit_card_id; --> NOT SURE IF YOU CAN UPDATE A FK
    
    /* INSERT block #11 - order_table_dw (from online_order) */
    INSERT INTO order_table_dw (order_id, inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id, data_source)
    SELECT order_id, inventory_id, item_name, item_price, item_quantity, date_of_purch, credit_card_id, 'online' AS data_source
    FROM online_order_table oo
    WHERE NOT EXISTS (
        SELECT order_id
        FROM order_table_dw ot
        WHERE ot.order_id = oo.order_id
    );
    
    /* UPDATE block #11 - order_table_dw (from online_order) */
    MERGE INTO order_table_dw odw
        USING online_order_table o
        ON (odw.order_id = o.order_id)
        WHEN MATCHED THEN
            UPDATE SET
                odw.inventory_id = o.inventory_id, --> NOT SURE IF YOU CAN UPDATE A FK
                odw.item_name = o.item_name,
                odw.item_price = o.item_price,
                odw.item_quantity = o.item_quantity,
                odw.date_of_purch = o.date_of_purch,
                odw.credit_card_id = o.credit_card_id; --> NOT SURE IF YOU CAN UPDATE A FK

END;
/
EXEC costco_etl_proc;


/* Step 4: Select a subset of data to create a data lake */
DROP VIEW join_one;
DROP VIEW for_export;

CREATE VIEW join_one AS
(SELECT v.vendor_id, vendor_name, street, city, state_name, country, zip, phone_number, invoice_id, invoice_number, invoice_date, invoice_total, payment_total, credit_total, invoice_due_date, payment_date
FROM vendor_table_dw v
JOIN invoice_table i
ON v.vendor_id = i.vendor_id);

CREATE OR REPLACE VIEW for_export
AS
(SELECT j.vendor_id, vendor_name, street, city, state_name, zip, phone_number, invoice_id, invoice_number, invoice_date, invoice_total, payment_total, credit_total, invoice_due_date, payment_date, inventory_id, item, cost_p_unit, quantity, low_threshold
FROM join_one j
FULL JOIN costco_inventory_table_dw c
ON j.vendor_id = c.vendor_id);
