CREATE TABLE `sof`.`posts` 
( `Id` INT NOT NULL , 
	`PostTypeId` INT NOT NULL , 
	`AcceptedAnswerId` INT NULL , 
	`ParentId` INT NULL , 
	`CreationDate` BIGINT NULL , 
	`Score` INT NULL , 
	`ViewCount` INT NULL , 
	`Body` TEXT CHARACTER SET utf8 COLLATE utf8_general_ci NULL , 
	`OwnerUserId` INT NULL , 
	`LastEditorUserId` INT NULL , 
	`LastEditorDisplayName` TEXT CHARACTER SET utf8 COLLATE utf8_general_ci NULL ,
	`LastEditDate` BIGINT NULL ,
	`LastActivityDate` BIGINT NULL , 
	`Title` TEXT CHARACTER SET utf8 COLLATE utf8_general_ci NULL , 
	`Tags` TEXT NULL , `AnswerCount` INT NULL , 
	`CommentCount` INT NULL , 
	`FavoriteCount` INT NULL , `CommunityOwnedDate` BIGINT NULL ,
	 PRIMARY KEY (`Id`)
) ENGINE = InnoDB;


CREATE TABLE `sof`.`tags` ( 
	`p_id` INT NOT NULL , 
	`tag` VARCHAR(150) NOT NULL , 
	PRIMARY KEY (`p_id`, `tag`)
) ENGINE = InnoDB;
ALTER TABLE `tags` ADD FOREIGN KEY (`p_id`) REFERENCES `posts`(`Id`) ON DELETE CASCADE ON UPDATE CASCADE;