SET @@session.long_query_time = 1000;
DROP TABLE IF EXISTS `temp_relations_TESTCORPUS2`;
CREATE TABLE `temp_relations_TESTCORPUS2` (
   `id` int(11) NOT NULL DEFAULT 0,
   `head` int(11) NOT NULL DEFAULT 0,
   `rel` ENUM('SS', 'OBJ', 'ADV', 'AA', 'AT', 'ET', 'PA') NOT NULL DEFAULT 'SS',
   `dep` int(11) NOT NULL DEFAULT 0,
   `freq` int(11) NOT NULL DEFAULT 0,
   `bfhead` BOOL  DEFAULT NULL,
   `bfdep` BOOL  DEFAULT NULL,
   `wfhead` BOOL  DEFAULT NULL,
   `wfdep` BOOL  DEFAULT NULL,
 PRIMARY KEY (`head`, `wfhead`, `dep`, `rel`, `freq`, `id`),
 INDEX `dep-wfdep-head-rel-freq-id` (`dep`, `wfdep`, `head`, `rel`, `freq`, `id`),
 INDEX `head-dep-bfhead-bfdep-rel-freq-id` (`head`, `dep`, `bfhead`, `bfdep`, `rel`, `freq`, `id`),
 INDEX `dep-head-bfhead-bfdep-rel-freq-id` (`dep`, `head`, `bfhead`, `bfdep`, `rel`, `freq`, `id`))  default charset = utf8mb4  row_format = compressed ;
DROP TABLE IF EXISTS `temp_relations_TESTCORPUS2_strings`;
CREATE TABLE `temp_relations_TESTCORPUS2_strings` (
   `id` int(11) NOT NULL DEFAULT 0,
   `string` varchar(100) NOT NULL DEFAULT '',
   `stringextra` varchar(32) NOT NULL DEFAULT '',
   `pos` varchar(5) NOT NULL DEFAULT '',
 PRIMARY KEY (`string`, `id`, `pos`, `stringextra`),
 INDEX `id-string-pos-stringextra` (`id`, `string`, `pos`, `stringextra`))  default charset = utf8mb4  collate = utf8mb4_bin  row_format = compressed ;
DROP TABLE IF EXISTS `temp_relations_TESTCORPUS2_rel`;
CREATE TABLE `temp_relations_TESTCORPUS2_rel` (
   `rel` ENUM('SS', 'OBJ', 'ADV', 'AA', 'AT', 'ET', 'PA') NOT NULL DEFAULT 'SS',
   `freq` int(11) NOT NULL DEFAULT 0,
 PRIMARY KEY (`rel`, `freq`))  default charset = utf8mb4  collate = utf8mb4_bin  row_format = compressed ;
DROP TABLE IF EXISTS `temp_relations_TESTCORPUS2_head_rel`;
CREATE TABLE `temp_relations_TESTCORPUS2_head_rel` (
   `head` int(11) NOT NULL DEFAULT 0,
   `rel` ENUM('SS', 'OBJ', 'ADV', 'AA', 'AT', 'ET', 'PA') NOT NULL DEFAULT 'SS',
   `freq` int(11) NOT NULL DEFAULT 0,
 PRIMARY KEY (`head`, `rel`, `freq`))  default charset = utf8mb4  collate = utf8mb4_bin  row_format = compressed ;
DROP TABLE IF EXISTS `temp_relations_TESTCORPUS2_dep_rel`;
CREATE TABLE `temp_relations_TESTCORPUS2_dep_rel` (
   `dep` int(11) NOT NULL DEFAULT 0,
   `rel` ENUM('SS', 'OBJ', 'ADV', 'AA', 'AT', 'ET', 'PA') NOT NULL DEFAULT 'SS',
   `freq` int(11) NOT NULL DEFAULT 0,
 PRIMARY KEY (`dep`, `rel`, `freq`))  default charset = utf8mb4  collate = utf8mb4_bin  row_format = compressed ;
DROP TABLE IF EXISTS `temp_relations_TESTCORPUS2_sentences`;
CREATE TABLE `temp_relations_TESTCORPUS2_sentences` (
   `id` int(11)  DEFAULT NULL,
   `sentence` varchar(64) NOT NULL DEFAULT '',
   `start` int(11)  DEFAULT NULL,
   `end` int(11)  DEFAULT NULL,
 INDEX `id` (`id`))  default charset = utf8mb4  collate = utf8mb4_bin  row_format = compressed ;
ALTER TABLE `temp_relations_TESTCORPUS2` DISABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_strings` DISABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_rel` DISABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_head_rel` DISABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_dep_rel` DISABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_sentences` DISABLE KEYS;
SET FOREIGN_KEY_CHECKS = 0;
SET UNIQUE_CHECKS = 0;
SET AUTOCOMMIT = 0;
SET NAMES utf8mb4;
INSERT INTO `temp_relations_TESTCORPUS2_strings` (id, pos, string, stringextra) VALUES
(1, '', '', ''),
(0, 'VB', 'börja..vb.1', ''),
(7, 'VB', 'börja..vb.2', ''),
(8, 'VB', 'börjar', ''),
(2, 'AB', 'här', ''),
(3, 'AB', 'här..ab.1', ''),
(9, 'VB', 'innehålla..vb.1', ''),
(20, 'VB', 'innehålla..vb.1', ''),
(12, 'VB', 'innehåller', ''),
(21, 'VB', 'innehåller', ''),
(13, 'VB', 'komma..vb.1', ''),
(14, 'VB', 'kommer', ''),
(26, 'AB', 'länge..ab.1', ''),
(27, 'AB', 'längre', ''),
(28, 'AB', 'längre..ab.1', ''),
(29, 'AB', 'lång..av.1', ''),
(15, 'NN', 'mening', ''),
(10, 'NN', 'mening', ''),
(17, 'NN', 'mening..nn.1', ''),
(11, 'NN', 'mening..nn.1', ''),
(19, 'NN', 'styck..nn.1', ''),
(4, 'NN', 'styck..nn.1', ''),
(22, 'NN', 'stycke..nn.1', ''),
(5, 'NN', 'stycke..nn.1', ''),
(23, 'NN', 'stycket', ''),
(6, 'NN', 'stycket', ''),
(24, 'VB', 'var', ''),
(25, 'VB', 'vara..vb.1', ''),
(16, 'VB', 'vara..vb.1', ''),
(30, 'VB', 'är', ''),
(18, 'VB', 'är', '');
INSERT INTO `temp_relations_TESTCORPUS2` (bfdep, bfhead, dep, freq, head, id, rel, wfdep, wfhead) VALUES
(0, 1, 2, 1, 0, 1, 'ADV', 1, 0),
(1, 1, 3, 1, 0, 2, 'ADV', 0, 0),
(1, 1, 1, 2, 0, 0, 'OBJ', 1, 0),
(1, 1, 4, 1, 0, 3, 'SS', 0, 0),
(1, 1, 5, 1, 0, 4, 'SS', 0, 0),
(0, 1, 6, 1, 0, 5, 'SS', 1, 0),
(0, 1, 2, 1, 7, 7, 'ADV', 1, 0),
(1, 1, 3, 1, 7, 8, 'ADV', 0, 0),
(1, 1, 1, 2, 7, 6, 'OBJ', 1, 0),
(1, 1, 4, 1, 7, 9, 'SS', 0, 0),
(1, 1, 5, 1, 7, 10, 'SS', 0, 0),
(0, 1, 6, 1, 7, 11, 'SS', 1, 0),
(1, 0, 3, 1, 8, 13, 'ADV', 0, 1),
(1, 0, 1, 1, 8, 12, 'OBJ', 0, 1),
(1, 0, 4, 1, 8, 14, 'SS', 0, 1),
(1, 0, 5, 1, 8, 15, 'SS', 0, 1),
(0, 1, 10, 1, 9, 16, 'OBJ', 1, 0),
(1, 1, 11, 1, 9, 17, 'OBJ', 0, 0),
(1, 0, 11, 1, 12, 18, 'OBJ', 0, 1),
(0, 1, 2, 1, 13, 20, 'ADV', 1, 0),
(1, 1, 3, 1, 13, 21, 'ADV', 0, 0),
(1, 1, 1, 2, 13, 19, 'OBJ', 1, 0),
(0, 1, 10, 1, 13, 22, 'SS', 1, 0),
(1, 1, 11, 1, 13, 23, 'SS', 0, 0),
(1, 0, 3, 1, 14, 25, 'ADV', 0, 1),
(1, 0, 1, 1, 14, 24, 'OBJ', 0, 1),
(1, 0, 11, 1, 14, 26, 'SS', 0, 1),
(1, 0, 16, 1, 15, 27, 'ET', 0, 1),
(1, 1, 16, 1, 17, 28, 'ET', 0, 0),
(0, 1, 18, 1, 17, 29, 'ET', 1, 0),
(1, 1, 20, 1, 19, 30, 'ET', 0, 0),
(0, 1, 21, 1, 19, 31, 'ET', 1, 0),
(1, 1, 20, 1, 22, 32, 'ET', 0, 0),
(0, 1, 21, 1, 22, 33, 'ET', 1, 0),
(1, 0, 20, 1, 23, 34, 'ET', 0, 1),
(1, 0, 1, 1, 24, 35, 'OBJ', 0, 1),
(1, 1, 26, 1, 25, 37, 'ADV', 0, 0),
(0, 1, 27, 1, 25, 38, 'ADV', 1, 0),
(1, 1, 28, 1, 25, 39, 'ADV', 0, 0),
(1, 1, 29, 1, 25, 40, 'ADV', 0, 0),
(1, 1, 1, 6, 25, 36, 'OBJ', 1, 0),
(1, 0, 26, 1, 30, 42, 'ADV', 0, 1),
(1, 0, 28, 1, 30, 43, 'ADV', 0, 1),
(1, 0, 29, 1, 30, 44, 'ADV', 0, 1),
(1, 0, 1, 2, 30, 41, 'OBJ', 0, 1);
INSERT INTO `temp_relations_TESTCORPUS2_rel` (freq, rel) VALUES
(6, 'ADV'),
(3, 'ET'),
(7, 'OBJ'),
(5, 'SS');
INSERT INTO `temp_relations_TESTCORPUS2_head_rel` (freq, head, rel) VALUES
(1, 0, 'ADV'),
(1, 0, 'OBJ'),
(2, 0, 'SS'),
(1, 7, 'ADV'),
(1, 7, 'OBJ'),
(2, 7, 'SS'),
(1, 8, 'ADV'),
(1, 8, 'OBJ'),
(2, 8, 'SS'),
(1, 9, 'OBJ'),
(1, 12, 'OBJ'),
(1, 13, 'ADV'),
(1, 13, 'OBJ'),
(1, 13, 'SS'),
(1, 14, 'ADV'),
(1, 14, 'OBJ'),
(1, 14, 'SS'),
(1, 15, 'ET'),
(1, 17, 'ET'),
(1, 19, 'ET'),
(1, 22, 'ET'),
(1, 23, 'ET'),
(1, 24, 'OBJ'),
(3, 25, 'ADV'),
(3, 25, 'OBJ'),
(3, 30, 'ADV'),
(2, 30, 'OBJ');
INSERT INTO `temp_relations_TESTCORPUS2_dep_rel` (dep, freq, rel) VALUES
(1, 12, 'OBJ'),
(2, 3, 'ADV'),
(3, 3, 'ADV'),
(4, 2, 'SS'),
(5, 2, 'SS'),
(6, 2, 'SS'),
(10, 1, 'OBJ'),
(10, 1, 'SS'),
(11, 1, 'OBJ'),
(11, 1, 'SS'),
(16, 1, 'ET'),
(18, 1, 'ET'),
(20, 2, 'ET'),
(21, 2, 'ET'),
(26, 1, 'ADV'),
(27, 1, 'ADV'),
(28, 1, 'ADV'),
(29, 1, 'ADV');
INSERT INTO `temp_relations_TESTCORPUS2_sentences` (end, id, sentence, start) VALUES
(2, 0, '8f2', 2),
(1, 1, '8f2', 2),
(1, 2, '8f2', 2),
(5, 3, '8f2', 2),
(5, 4, '8f2', 2),
(5, 5, '8f2', 2),
(2, 6, '8f2', 2),
(1, 7, '8f2', 2),
(1, 8, '8f2', 2),
(5, 9, '8f2', 2),
(5, 10, '8f2', 2),
(5, 11, '8f2', 2),
(2, 12, '8f2', 2),
(1, 13, '8f2', 2),
(5, 14, '8f2', 2),
(5, 15, '8f2', 2),
(10, 16, '8f2', 7),
(10, 17, '8f2', 7),
(10, 18, '8f2', 7),
(2, 19, '80d', 2),
(1, 20, '80d', 2),
(1, 21, '80d', 2),
(5, 22, '80d', 2),
(5, 23, '80d', 2),
(2, 24, '80d', 2),
(1, 25, '80d', 2),
(5, 26, '80d', 2),
(7, 27, '80d', 5),
(7, 28, '80d', 5),
(7, 29, '80d', 5),
(7, 30, '8f2', 5),
(7, 31, '8f2', 5),
(7, 32, '8f2', 5),
(7, 33, '8f2', 5),
(7, 34, '8f2', 5),
(2, 35, '838', 2),
(7, 36, '80d', 7),
(2, 36, '838', 2),
(3, 36, '86a', 3),
(9, 37, '80d', 7),
(9, 38, '80d', 7),
(9, 39, '80d', 7),
(9, 40, '80d', 7),
(7, 41, '80d', 7),
(3, 41, '86a', 3),
(9, 42, '80d', 7),
(9, 43, '80d', 7),
(9, 44, '80d', 7);
ALTER TABLE `temp_relations_TESTCORPUS2` ENABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_strings` ENABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_rel` ENABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_head_rel` ENABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_dep_rel` ENABLE KEYS;
ALTER TABLE `temp_relations_TESTCORPUS2_sentences` ENABLE KEYS;
DROP TABLE IF EXISTS `relations_TESTCORPUS2`, `relations_TESTCORPUS2_strings`, `relations_TESTCORPUS2_rel`, `relations_TESTCORPUS2_head_rel`, `relations_TESTCORPUS2_dep_rel`, `relations_TESTCORPUS2_sentences`;
RENAME TABLE `temp_relations_TESTCORPUS2` TO `relations_TESTCORPUS2`, `temp_relations_TESTCORPUS2_strings` TO `relations_TESTCORPUS2_strings`, `temp_relations_TESTCORPUS2_rel` TO `relations_TESTCORPUS2_rel`, `temp_relations_TESTCORPUS2_head_rel` TO `relations_TESTCORPUS2_head_rel`, `temp_relations_TESTCORPUS2_dep_rel` TO `relations_TESTCORPUS2_dep_rel`, `temp_relations_TESTCORPUS2_sentences` TO `relations_TESTCORPUS2_sentences`;
SET UNIQUE_CHECKS = 1;
SET FOREIGN_KEY_CHECKS = 1;
COMMIT;
