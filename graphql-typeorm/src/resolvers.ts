import { User } from "./entity/User";
import { BrainTumorImage } from "./entity/BrainTumorImage";
import { getConnection } from "typeorm";
import bcrypt from "bcrypt";

// Provide resolver functions for your schema fields
export const resolvers = {
  Query: {
    getUser: async (_: any, args: any, { user, alwaysAuthenticate }: any) => {
      const { filter } = args;
      if (!user && !alwaysAuthenticate) {
        throw new Error("unauthenticated!");
      }

      if (filter.emails) {
        return await User.findOne({ where: { email: filter.emails[0] } });
      } else if (filter.ids) {
        return await User.findOne({ where: { id: filter.ids[0] } });
      } else {
        return false;
      }
    },
    verifyPassword: async (_: any, args: any, { alwaysAuthenticate }: any) => {
      if (!alwaysAuthenticate) {
        throw new Error("unauthenticated!");
      }

      const { filter, password } = args;

      let user: User | undefined = undefined;
      if (filter.emails) {
        user = await User.findOne({ where: { email: filter.emails[0] } });
      } else if (filter.ids) {
        user = await User.findOne({ where: { id: filter.ids[0] } });
      }

      if (!user) {
        return false;
      }

      const hashPass = user.password;

      return await bcrypt.compare(password, hashPass);
    },
    getRandomImage: async () => {
      return await BrainTumorImage.getRandomImage();
    },
  },
  Mutation: {
    addUser: async (_: any, args: any, { alwaysAuthenticate }: any) => {
      if (!alwaysAuthenticate) {
        throw new Error("unauthenticated!");
      }

      const { firstName, lastName, age, username, email, password } = args;

      const saltRounds = 13;
      let hashPass = undefined;

      await bcrypt
        .hash(password, saltRounds)
        .then((hash) => {
          hashPass = hash;
        })
        .catch((err) => {
          console.log(err);
          return false;
        });

      try {
        const user = User.create({
          firstName,
          lastName,
          age,
          username,
          email,
          password: hashPass,
        });

        await user.save();
        console.log("Saved a new user with id: " + user.id);

        return true;
      } catch (error) {
        return false;
      }
    },
  },
};
