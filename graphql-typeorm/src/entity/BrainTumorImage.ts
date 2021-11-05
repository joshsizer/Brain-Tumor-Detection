import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  BaseEntity,
  getConnection,
} from "typeorm";

@Entity({
  name: "brain_tumor_image",
})
export class BrainTumorImage extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: "text", nullable: false, unique: true })
  path: string;

  @Column({ type: "text", nullable: false })
  classification: string;

  @Column({ type: "integer", nullable: false })
  width: number;

  @Column({ type: "integer", nullable: false })
  height: number;

  static async getRandomImage(): Promise<BrainTumorImage | undefined> {
    return getConnection()
      .getRepository(BrainTumorImage)
      .createQueryBuilder()
      .orderBy("RANDOM()")
      .limit(1)
      .getOne();
  }
}
