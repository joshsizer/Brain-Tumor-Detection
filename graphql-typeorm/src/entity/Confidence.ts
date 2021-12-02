import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  BaseEntity,
  getConnection,
} from "typeorm";

@Entity({
  name: "confidence",
})
export class Confidence extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: "integer", nullable: false })
  index: number;

  @Column({ type: "integer", nullable: false })
  count: number;
}
