DROP DATABASE IF EXISTS `dimentia`;
CREATE DATABASE `dimentia`;
USE `dimentia`;

CREATE TABLE `users` (
    `id` INT PRIMARY KEY AUTO_INCREMENT, 
    `name` VARCHAR(1000),
    `email` VARCHAR(1000),
    `password` VARCHAR(225)
);

CREATE TABLE `relatives` (
    `id` INT PRIMARY KEY AUTO_INCREMENT,
    `name` VARCHAR(1000),
    `img` LONGBLOB,
    `relation` VARCHAR(1000),
    `description` VARCHAR(1000),
    `audio` LONGBLOB,
    `patient_id` INT,
    `mobile` VARCHAR(15)
);

CREATE TABLE `patient_condition` (
    `id` INT PRIMARY KEY AUTO_INCREMENT,
    `img` LONGBLOB,
    `condition` VARCHAR(1000),
    `description` VARCHAR(1000),
    `patient_id` INT
);


