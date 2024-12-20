"""Added user

Revision ID: 56ffa939fea8
Revises: 3b52fcd72885
Create Date: 2024-11-06 14:33:26.770597

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '56ffa939fea8'
down_revision = '3b52fcd72885'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_approved', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('role', sa.String(length=20), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_column('role')
        batch_op.drop_column('is_approved')

    # ### end Alembic commands ###
