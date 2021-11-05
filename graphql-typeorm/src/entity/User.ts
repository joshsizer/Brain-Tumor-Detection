import { Entity, PrimaryGeneratedColumn, Column, BaseEntity } from "typeorm";

@Entity({
  name: "member",
})
export class User extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: "text", nullable: true })
  firstName: string | null;

  @Column({ type: "text", nullable: true })
  lastName: string | null;

  @Column({ type: "integer", nullable: true })
  age: number | null;

  @Column({ type: "text", nullable: false, unique: true })
  username: string;

  @Column({ type: "text", nullable: false, unique: true })
  email: string;

  @Column({ type: "text", nullable: false })
  password: string;

  @Column({ default: 0 })
  refreshCount: number;
}
